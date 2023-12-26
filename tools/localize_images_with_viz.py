import dataclasses
import os
import pickle
from typing import Type, Dict, List, Tuple, Optional
import pangolin
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from OpenGL import GL as gl

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from tools.features.base import KeypointMatchingResults, KeypointData
from tools.features.outlier_filter import keep_inliers, outlier_filtering, FilterType
from tools.features.superglue import SuperGlue
from tools.features.superpoint import SuperPoint
from tools.vpr.encode_dataset import VprEncoder
from tools.vpr.options import options
from tools.vpr.utils.vpr_utils import get_nearest_neighbors
from tools.vpr_scenarios import Scenario_2023_11_27T90_37_23, VprScenario

from gtsam import Pose3, Rot3, Cal3_S2, PinholeCameraCal3_S2
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = VprEncoder(encoder_type='cosplace',
                     encoder_options=options['cosplace'],
                     device_id=device)


@dataclasses.dataclass
class KeepInliersResult:
    matches: List[cv2.DMatch]
    kpts_query: List[cv2.KeyPoint]
    kpts_train: List[cv2.KeyPoint]

    @staticmethod
    def parse_matches_dict(data: Dict) -> List[cv2.DMatch]:
        return [cv2.DMatch(queryIdx, trainIdx, distance) for queryIdx, trainIdx, distance in zip(data['queryIdx'], data['trainIdx'], data['distance'])]

    @staticmethod
    def parse_kepoints_dict(data: Dict) -> List[cv2.KeyPoint]:
        return [cv2.KeyPoint(pt_x, pt_y, size, angle, response, octave, class_id) for pt_x, pt_y, size, angle, response, octave, class_id in zip(data['pt_x'], data['pt_y'], data['size'], data['angle'], data['response'], data['octave'], data['class_id'])]

    def get_matches_as_dict(self) -> Dict:
        return {"queryIdx": [m.queryIdx for m in self.matches],
                "trainIdx": [m.trainIdx for m in self.matches],
                "distance": [m.distance for m in self.matches]}

    def get_kpts_as_dict(self, kpts: List[cv2.KeyPoint]) -> Dict:
        return {"pt_x": [kpt.pt[0] for kpt in kpts],
                "pt_y": [kpt.pt[1] for kpt in kpts],
                "size": [kpt.size for kpt in kpts],
                "angle": [kpt.angle for kpt in kpts],
                "response": [kpt.response for kpt in kpts],
                "octave": [kpt.octave for kpt in kpts],
                "class_id": [kpt.class_id for kpt in kpts]}

    @classmethod
    def from_dict(cls, data: Dict):
        return KeepInliersResult(matches=cls.parse_matches_dict(data['matches']),
                                 kpts_query=cls.parse_kepoints_dict(data['kpts_query']),
                                 kpts_train=cls.parse_kepoints_dict(data['kpts_train']))

    def as_dict(self) -> Dict:
        return {"matches": self.get_matches_as_dict(),
                "kpts_query": self.get_kpts_as_dict(self.kpts_query),
                "kpts_train": self.get_kpts_as_dict(self.kpts_train)}


@dataclasses.dataclass
class OutlierFilteringResult:
    inliers: KeepInliersResult
    match_image: Optional[np.ndarray]

    def save_to_file(self, filename: str) -> Dict:
        data = {"inliers": self.inliers.as_dict(),
                "match_image": self.match_image}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_file(cls, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return OutlierFilteringResult(inliers=KeepInliersResult.from_dict(data['inliers']),
                                      match_image=data['match_image'])


def apply_outlier_filtering(kp_matching_res: KeypointMatchingResults, kp_query: KeypointData, kp_train: KeypointData, K: np.ndarray = None, query_image: np.ndarray = None,
                            train_image: np.ndarray = None, filter_type: FilterType = FilterType.HOMOGRAPHY) -> OutlierFilteringResult:
    kpts_train = np.array([kp_matching_res.train_keypoints[m.trainIdx].pt for m in kp_matching_res.matches])
    kpts_query = np.array([kp_matching_res.query_keypoints[m.queryIdx].pt for m in kp_matching_res.matches])
    mat, inliers = outlier_filtering(filter_type, kpts_train, kpts_query, ransac_inlier_thr=3, K=K)
    inlier_matches_idx = np.where(inliers)[0].astype(np.int32).tolist()
    inlier_matches = [kp_matching_res.matches[i] for i in inlier_matches_idx]

    inliers: KeepInliersResult = keep_inliers(inlier_matches, kp_matching_res.query_keypoints,
                                              kp_matching_res.train_keypoints)
    if query_image is None:
        query_image = kp_query.image
    if train_image is None:
        train_image = kp_train.image
    match_image: np.ndarray = cv2.drawMatches(query_image,
                                              inliers.kpts_query,
                                              train_image,
                                              inliers.kpts_train,
                                              inliers.matches,
                                              None)
    return OutlierFilteringResult(inliers=inliers, match_image=match_image)


def search_db(db: torch.Tensor, db_offset: int, target_encoding: torch.Tensor, label: str, results_path: Path, plot_flag: bool):
    search_db = db[db_offset:]
    similarity = torch.matmul(search_db, torch.unsqueeze(target_encoding, 0).transpose(0, 1)).squeeze() / ((torch.norm(search_db, dim=1)) * torch.norm(target_encoding))

    if plot_flag:
        plt.figure()
        plt.plot(np.arange(db_offset, db_offset + int(search_db.shape[0])), np.array(similarity))
        plt.xlabel('frames')
        plt.ylabel('similarity')
        plt.grid()
        plt.savefig(results_path / 'similarity_{0:d}_{1:s}.png'.format(TARGET_FRAME, label))

    N_ROWS: int = 3
    N_COLS: int = 3
    CENTER_PLOT_INDICES = ((N_ROWS - 1) // 2, (N_COLS - 1) // 2)
    PLOT_INDICES = set([(0, 0), (0, N_ROWS - 1), (N_COLS - 1, 0), (N_ROWS - 1, N_COLS - 1), CENTER_PLOT_INDICES])
    knn_indices, knn_scores = get_nearest_neighbors(query_vec=torch.unsqueeze(target_encoding, 0), db_vec=search_db, k=4, return_scores=True)
    knn_indices += db_offset

    if plot_flag:
        k: int = 0
        fig, axes = plt.subplots(N_ROWS, N_COLS)
        fig.set_figheight(15)
        fig.set_figwidth(20)
        for row in range(N_ROWS):
            for col in range(N_COLS):
                if (row, col) not in PLOT_INDICES:
                    continue
                elif (row, col) == CENTER_PLOT_INDICES:
                    axes[row, col].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
                    axes[row, col].set_title("target image ({0:d})".format(TARGET_FRAME))
                else:
                    img: np.ndarray = cv2.imread(str(images_path / "frame{0:d}.jpg".format(int(knn_indices[k]))))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title("{0:d}(similarity={1:.2f})".format(int(knn_indices[k]), float(knn_scores[k])))
                    k += 1
        plt.savefig(results_path / 'knn_{0:d}_{1:s}.png'.format(TARGET_FRAME, label))
    return knn_indices, knn_scores


class MatchHypothesis:
    frame_idx: int
    target_frame_idx: int
    matching_result: OutlierFilteringResult
    relative_pose_wrt_target_frame: np.ndarray

    def __init__(self, frame_idx: int, target_frame_index: int):
        self.frame_idx = frame_idx
        self.target_frame_idx = target_frame_index

@dataclasses.dataclass
class FrameData:
    match_with_ref: MatchHypothesis
    match_with_neighbor: MatchHypothesis

def estimate_relative_pose(match: MatchHypothesis, K: np.ndarray):
    kp_ref: KeypointData = np.array([k.pt for k in match.matching_result.inliers.kpts_train])
    kp_img: KeypointData = np.array([k.pt for k in match.matching_result.inliers.kpts_query])

    DEFAULT_NISTER_PROB: float = .999
    DEFAULT_NISTER_THRESHOLD: float = 2.0

    essential_mat, inliers = cv2.findEssentialMat(points1=kp_img,
                                            points2=kp_ref,
                                            cameraMatrix=K,
                                            method=cv2.RANSAC,
                                            prob=DEFAULT_NISTER_PROB,
                                            threshold=DEFAULT_NISTER_THRESHOLD)
    kp_img = np.array([kp for i, kp in enumerate(kp_img) if inliers[i]])
    kp_ref = np.array([kp for i, kp in enumerate(kp_ref) if inliers[i]])
    retval, R, t, mask = cv2.recoverPose(E=essential_mat,
                                         points1=kp_img,
                                         points2=kp_ref,
                                         cameraMatrix=K)
    rel_pose = np.eye(4)
    rel_pose[:3, :3] = R
    rel_pose[:3, 3] = t.squeeze()
    match.relative_pose_wrt_target_frame = rel_pose


def match_pair(match: MatchHypothesis, images_path: Path, plot_flag: bool, save_flag: bool, device: torch.device, results_path: str = None, K: np.ndarray = None, filter_type: FilterType = FilterType.HOMOGRAPHY):
    ref_img: np.ndarray = cv2.imread(str(images_path / "frame{0:d}.jpg".format(match.target_frame_idx)), cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.resize(ref_img, [ref_img.shape[1] // 2, ref_img.shape[0] // 2])
    img: np.ndarray = cv2.imread(str(images_path / "frame{0:d}.jpg".format(match.frame_idx)), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [img.shape[1] // 2, img.shape[0] // 2])

    superglue = SuperGlue({'weights': 'outdoor'}).to(device).eval()
    superpoint = SuperPoint({}).to(device).eval()

    kp_ref: KeypointData = superpoint.detectAndCompute(ref_img, device)
    kp_img: KeypointData = superpoint.detectAndCompute(img, device)
    kp_matching_res: KeypointMatchingResults = superglue.match(kp_img, kp_ref)

    matches_after_filter: OutlierFilteringResult = apply_outlier_filtering(kp_matching_res=kp_matching_res, kp_query=kp_img, kp_train=kp_ref, filter_type=filter_type, K=K)
    match.matching_result = matches_after_filter

    os.makedirs(results_path / "local_matching", exist_ok=True)
    if plot_flag:
        cv2.imshow('loop-closure hypothesis matches ({0:d}<>{1:d})'.format(match.target_frame_idx, match.frame_idx), matches_after_filter.match_image)
    if save_flag:
        cv2.imwrite(str(results_path / "local_matching" / "match_to_ref_{0:d}_{1:d}.png".format(match.target_frame_idx, match.frame_idx)), matches_after_filter.match_image)


def to_gtsam_pose(pose: np.ndarray) -> Pose3:
    return Pose3(Rot3(pose[:3, :3]), pose[:3, 3])


def plot_poses(fig_num: int, poses: List[np.ndarray], title: str):
    fig = plt.figure(fig_num, figsize=(10, 10))
    axes: Axes3D = fig.add_subplot(projection='3d', proj_type='ortho')
    axes.view_init(.0, 90.0)
    for cam_pose in poses:
        gtsam_plot.plot_pose3(fig_num, to_gtsam_pose(cam_pose), axis_length=10)
        gtsam_plot.plot_point3(fig_num, cam_pose[:3, 3], linespec='go')
    axes.set_xlim3d(-7, 7)
    axes.set_ylim3d(-7, 7)
    axes.set_zlim3d(-7, 7)
    axes.set_title(title)
    gtsam_plot.set_axes_equal(fig_num)

class PangolinModelViewParams:
    eye_position_x: float
    eye_position_y: float
    eye_position_z: float
    lookat_x: float
    lookat_y: float
    lookat_z: float
    up_vector_x: float
    up_vector_y: float
    up_vector_z: float

    def __init__(self, eye_position_x: float,
                 eye_position_y: float,
                 eye_position_z: float,
                 lookat_x: float,
                 lookat_y: float,
                 lookat_z: float,
                 up_vector_x: float,
                 up_vector_y: float,
                 up_vector_z: float):
        self.eye_position_x = eye_position_x
        self.eye_position_y = eye_position_y
        self.eye_position_z = eye_position_z
        self.lookat_x = lookat_x
        self.lookat_y = lookat_y
        self.lookat_z = lookat_z
        self.up_vector_x = up_vector_x
        self.up_vector_y = up_vector_y
        self.up_vector_z = up_vector_z

@dataclasses.dataclass
class ObjectBboxParams:
    center: np.ndarray
    length_x: float
    length_y: float
    length_z: float
    heading: float
def draw_pointclouds_using_pangolin(list_of_point_clouds: List[np.ndarray],
                                    list_of_camera_poses: List[np.ndarray],
                                    model_view_params: PangolinModelViewParams = None,
                                    ground_plane_resolution: float = 1.0,
                                    ground_plane_axes: str = 'xz',
                                    num_divs: int = 200,
                                    label: str = 'Main',
                                    save_path: str = None,
                                    point_size: int = 5,
                                    object_bbox_params: ObjectBboxParams = None,
                                    list_of_camera_colors: List[tuple] = None,
                                    background_color: Tuple[float] = None):
    def drawPlane(num_divs=200, div_size=1.0, ground_plane_axes: str = 'xz'):
        # Plane parallel to x-z at origin with normal -y
        if ground_plane_axes is None:
            return

        minx = -num_divs * div_size
        maxx = num_divs * div_size
        if ground_plane_axes == 'xz':
            minz = -num_divs * div_size
            maxz = num_divs * div_size
        elif ground_plane_axes == 'xy':
            miny = -num_divs * div_size
            maxy = num_divs * div_size

        # gl.glLineWidth(2)
        # gl.glColor3f(0.7,0.7,1.0)

        gl.glBegin(gl.GL_LINES)
        for n in range(2 * num_divs):
            if ground_plane_axes == 'xz':
                gl.glVertex3f(minx + div_size * n, 0, minz)
                gl.glVertex3f(minx + div_size * n, 0, maxz)
                gl.glVertex3f(minx, 0, minz + div_size * n)
                gl.glVertex3f(maxx, 0, minz + div_size * n)
            elif ground_plane_axes == 'xy':
                gl.glVertex3f(minx + div_size * n, miny, 0)
                gl.glVertex3f(minx + div_size * n, maxy, 0)
                gl.glVertex3f(minx, miny + div_size * n, 0)
                gl.glVertex3f(maxx, miny + div_size * n, 0)
        gl.glEnd()

    pangolin.CreateWindowAndBind(label, 1920, 1080)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    if model_view_params is None:
        model_look_at: pangolin.ModelViewLookAt = pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0,
                                                                           pangolin.AxisDirection.AxisY)
    else:
        model_look_at: pangolin.ModelViewLookAt = pangolin.ModelViewLookAt(model_view_params.eye_position_x if object_bbox_params is None else object_bbox_params.center[0],
                                                                           model_view_params.eye_position_y if object_bbox_params is None else object_bbox_params.center[1],
                                                                           model_view_params.eye_position_z if object_bbox_params is None else object_bbox_params.center[2] - 10.0,
                                                                           model_view_params.lookat_x if object_bbox_params is None else object_bbox_params.center[0],
                                                                           model_view_params.lookat_y if object_bbox_params is None else object_bbox_params.center[1],
                                                                           model_view_params.lookat_z if object_bbox_params is None else object_bbox_params.center[2],
                                                                           model_view_params.up_vector_x,
                                                                           model_view_params.up_vector_y,
                                                                           model_view_params.up_vector_z)

    scam = pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200), model_look_at)
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
    dcam.SetHandler(handler)

    while not pangolin.ShouldQuit():
        np.random.seed(200)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if background_color is None:
            background_color = (1.0, 1.0, 1.0)
        gl.glClearColor(background_color[0], background_color[1], background_color[2], 1.0)

        dcam.Activate(scam)

        drawPlane(div_size=ground_plane_resolution,
                  num_divs=num_divs,
                  ground_plane_axes=ground_plane_axes)

        gl.glLineWidth(10)
        gl.glColor3f(0.0, 0.0, 0.0)
        traj_x: np.ndarray = np.array([[.0, .0, .0], [1.0, .0, .0]])
        traj_y: np.ndarray = np.array([[.0, .0, .0], [.0, 1.0, .0]])
        traj_z: np.ndarray = np.array([[.0, .0, .0], [.0, .0, 1.0]])
        gl.glColor3f(1.0, .0, 0.0)
        pangolin.DrawLine(traj_x)  # consecutive
        gl.glColor3f(.0, 1.0, 0.0)
        pangolin.DrawLine(traj_y)  # consecutive
        gl.glColor3f(.0, .0, 1.0)
        pangolin.DrawLine(traj_z)  # consecutive

        # Draw Point Cloud
        for pcl_idx, pcl in enumerate(list_of_point_clouds):
            points: np.ndarray = pcl
            # points = points * 3 + 1
            # if pcl.point_size is not None:
            #     gl.glPointSize(pcl.point_size)
            # else:
            gl.glPointSize(point_size)

            # if pcl.color is not None:
            #     pangolin.DrawPoints(points, pcl.color)
            # else:
            gl.glColor3f(.0, .0 + pcl_idx / len(list_of_point_clouds), .0)
            pangolin.DrawPoints(points)

        # Draw ego-vehicle
        poses = [np.identity(4)]
        vehicle_length: float = 5.0
        vehicle_width: float = 2.0
        vehicle_height: float = 2.0
        sizes = np.array([[vehicle_width, vehicle_height, vehicle_length]])
        gl.glLineWidth(1)
        gl.glColor3f(1.0, 0.0, 1.0)
        pangolin.DrawBoxes(poses, sizes)

        if object_bbox_params is not None:
            sizes = np.array([[object_bbox_params.length_x, object_bbox_params.length_y, object_bbox_params.length_z]])
            gl.glLineWidth(10)
            gl.glColor3f(1.0, .0, .0)
            bbox_pose: np.ndarray = np.eye(4)
            bbox_pose[:3, 3] = object_bbox_params.center
            bbox_pose[:3, :3] = Rotation.from_euler('YZX', [object_bbox_params.heading, .0, .0]).as_matrix()
            pangolin.DrawBoxes([bbox_pose], sizes)

        # Draw camera poses
        if list_of_camera_colors is None:
            list_of_camera_colors = [(0.0, 0.0, 1.0) for pose in list_of_camera_poses]
        for pose, cam_color in zip(list_of_camera_poses, list_of_camera_colors):
            gl.glLineWidth(3)
            gl.glColor3f(*cam_color)
            pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)
        if save_path is not None:
            pangolin.SaveWindowOnRender(save_path)

        pangolin.FinishFrame()

if __name__ == "__main__":
    # read encodings
    scenario: Type[VprScenario] = Scenario_2023_11_27T90_37_23()
    PLOT_FIGURES: bool = False
    EXPORT_FIGURES: bool = True

    K: np.ndarray = np.array([[1334.572266, 0, 960.029785],
                              [.0, 1334.572266, 728.388794],
                              [.0, .0, 1.0]])

    data_path: Path = scenario.data_path
    encodings_path: Path = data_path / 'encodings' / 'cosplace_small_res'
    images_path: Path = data_path / 'images'
    results_path: Path = data_path / 'results'
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(results_path / "local_matching", exist_ok=True)

    db_vec: torch.Tensor = torch.load(encodings_path / 'encodings.torch').to(device)
    TARGET_FRAME: int = scenario.entrance_good_frame
    target_img: np.ndarray = cv2.imread(str(images_path / "frame{0:d}.jpg".format(TARGET_FRAME)))
    cv2.imshow("target frame ({0:d})".format(TARGET_FRAME), target_img)

    ref_frame_index = TARGET_FRAME
    target_encoding: torch.Tensor = db_vec[ref_frame_index].to(device)
    db_offset: int = 8370
    nn, scores = search_db(db=db_vec, db_offset=db_offset, target_encoding=target_encoding, label='after_{0:d}'.format(db_offset), results_path=results_path, plot_flag=PLOT_FIGURES)

    # matching
    frame_pairs: List[FrameData] = []
    for frame_offset in np.arange(0, 30, 1):
        try:
            frame_idx: int = nn[0] + frame_offset
            # neighbor_idx: int = frame_idx if frame_offset == 0 else frame_idx - 1
            neighbor_idx: int = nn[0]
            match_with_ref: MatchHypothesis = MatchHypothesis(target_frame_index=ref_frame_index, frame_idx=frame_idx)
            match_pair(match_with_ref, images_path, plot_flag=PLOT_FIGURES, save_flag=EXPORT_FIGURES, device=device, results_path=results_path, K=K)

            match_with_neighbor: MatchHypothesis = MatchHypothesis(target_frame_index=neighbor_idx, frame_idx=frame_idx)
            match_pair(match_with_neighbor, images_path, plot_flag=PLOT_FIGURES, save_flag=EXPORT_FIGURES, device=device, results_path=results_path, K=K)

            frame_data: FrameData = FrameData(match_with_ref=match_with_ref,
                                              match_with_neighbor=match_with_neighbor)
            frame_pairs.append(frame_data)
        except:
            continue

    #pose estimation
    list_of_abs_poses_wrt_target: List[np.ndarray] = []
    list_of_accumulated_poses_wrt_target: List[np.ndarray] = []
    accumulated_pose = None
    for frame_pair in frame_pairs:
        estimate_relative_pose(frame_pair.match_with_ref, K=K)
        pose = frame_pair.match_with_ref.relative_pose_wrt_target_frame
        list_of_abs_poses_wrt_target.append(frame_pair.match_with_ref.relative_pose_wrt_target_frame)

        estimate_relative_pose(frame_pair.match_with_neighbor, K=K)
        pose = frame_pair.match_with_neighbor.relative_pose_wrt_target_frame
        if accumulated_pose is None:
            accumulated_pose = pose
        else:
            accumulated_pose = accumulated_pose @ pose
        list_of_accumulated_poses_wrt_target.append(pose)

    COLOR_1 = (255, 0, 0)
    COLOR_2 = (0, 255, 0)
    list_of_camera_poses = []
    list_of_point_clouds = []
    list_of_colors = []
    to_ref_pose = np.linalg.inv(list_of_abs_poses_wrt_target[0])
    to_ref_pose = np.eye(4)
    for pose in list_of_abs_poses_wrt_target:
        list_of_camera_poses.append(to_ref_pose @ pose)
        list_of_colors.append(COLOR_1)

    to_ref_pose = np.linalg.inv(list_of_accumulated_poses_wrt_target[0])
    to_ref_pose = np.eye(4)
    for pose in list_of_accumulated_poses_wrt_target:
        list_of_camera_poses.append(to_ref_pose @ pose)
        list_of_colors.append(COLOR_2)

    draw_pointclouds_using_pangolin(list_of_camera_poses=list_of_abs_poses_wrt_target,
                                    list_of_point_clouds=[],
                                    model_view_params=None,
                                    ground_plane_axes='xz',
                                    point_size=10,
                                    label='Absolute poses wrt target')

    draw_pointclouds_using_pangolin(list_of_camera_poses=list_of_accumulated_poses_wrt_target,
                                    list_of_point_clouds=[],
                                    model_view_params=None,
                                    ground_plane_axes='xz',
                                    point_size=10,
                                    label='Accumulated relative poses')

    # plot_poses(0, list_of_abs_poses_wrt_target, 'abs poses wrt target frame')
    # plot_poses(1, list_of_accumulated_poses_wrt_target, 'accumulated poses')
    plt.show(block=True)

    cv2.waitKey(1000)
    if PLOT_FIGURES:
        plt.show()
