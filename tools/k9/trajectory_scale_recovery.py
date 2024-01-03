import glob
from pathlib import Path
from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from tools.pangolin_viz import Viewer

"""
#camera matrix:
# [[720.74677588   0.         324.72460102]
# [  0.         720.87434544 238.97483313]
# [  0.           0.           1.        ]]
#distortion coefficients:  [-1.96841029e-02 -5.74293178e-01 -6.78065914e-04  8.34984375e-04 1.59969276e+00]
pattern_size = (8 , 6)
args.setdefault('--square_size', 2.714)
"""

def get_frame_id(name: str) -> int:
    return int(name.split('.')[0].split('frame')[1])

def resize(img: np.ndarray, factor: int = 1):
    if factor == 1:
        return img
    else:
        return cv2.resize(img, [img.shape[1]//factor, img.shape[0]//factor])

def solve_pnp(world_points: np.ndarray, object_points: np.ndarray, K: np.ndarray, distCoeffs: np.ndarray, max_reprojection: float) -> np.ndarray:
    (_, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(objectPoints=world_points,
                                                                           imagePoints=object_points,
                                                                           cameraMatrix=K,
                                                                           distCoeffs=distCoeffs,
                                                                           reprojectionError=max_reprojection)
    if inliers is None or len(inliers) < 3:
        return None, None

    rotation_matrix: np.ndarray = cv2.Rodrigues(rotation_vector)[0]
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector.squeeze()
    Tinv = np.linalg.inv(T)
    return Tinv, inliers

if __name__ == "__main__":
    IMAGE_DOWNSIZE_FACTOR: int = 3
    START_FRAME: int = 350
    END_FRAME: int = 400
    N_ROWS: int = 8
    N_COLS: int = 6
    PATTERN_SIZE = (N_ROWS, N_COLS)
    K = np.array([[1.78020320e+03, 0.00000000e+00, 1.93685914e+03],
                  [0.00000000e+00, 1.78309483e+03, 1.08289021e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    DIST_COEFFS = np.array([-2.56016339e-01, 8.39276935e-02, -3.34335209e-04, 5.82636414e-05, -1.43993861e-02])

    SQUARE_SIZE: float = 2.714*1e-2
    WORLD_POINTS = np.zeros((N_ROWS * N_COLS, 3), np.float32)
    WORLD_POINTS[:, :2] = SQUARE_SIZE * np.mgrid[0:N_ROWS, 0:N_COLS].T.reshape(-1, 2)
    MAX_REPROJECTION_ERR_PX: int = 5
    VISUALIZATION: bool = True

    K_downsize = K / IMAGE_DOWNSIZE_FACTOR
    K_downsize[2,2] = 1.0

    if VISUALIZATION:
        viewer: Viewer() = Viewer()
        cameras: List = []
        camera_pose = np.eye(4)
        cameras.append(camera_pose)

    images_path: Path = Path("/october23/tunnels/K9/calib/GX010909/images")
    list_of_image_paths: List[str] = sorted(glob.glob(str(images_path / '*.jpg')), key=lambda x: get_frame_id(x))

    if VISUALIZATION:
        accumulated_pose = np.eye(4)
        list_of_absolute_poses = [accumulated_pose]
        viewer.q_camera.put(list_of_absolute_poses)
        viewer.q_pose.put(accumulated_pose)

    distances_to_target: List[float] = []
    frame_id_for_plot: List[int] = []
    for image_path in tqdm(list_of_image_paths, total=len(list_of_image_paths)):
        frame_id: int = get_frame_id(image_path)
        if frame_id < START_FRAME or frame_id > END_FRAME:
            continue

        img = resize(cv2.imread(image_path), factor=IMAGE_DOWNSIZE_FACTOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_for_plot: np.ndarray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(image=gray,
                                             patternSize=PATTERN_SIZE,
                                             corners=None)
        if not ret:
            print("failed to find corners in image {0:d}".format(frame_id))
            continue

        frame_with_corners = cv2.drawChessboardCorners(gray_for_plot, PATTERN_SIZE, corners, patternWasFound=True)
        pose, inliers = solve_pnp(world_points=WORLD_POINTS,
                                  object_points=corners,
                                  K=K_downsize,
                                  distCoeffs=DIST_COEFFS,
                                  max_reprojection=MAX_REPROJECTION_ERR_PX)
        if pose is None or inliers is None:
            continue

        position_world: np.ndarray = pose[:3, 3]
        rotation_world: np.ndarray = pose[:3, :3]
        heading, pitch, roll = Rotation.from_matrix(rotation_world).as_euler('ZYX', degrees=True)
        if VISUALIZATION:
            list_of_absolute_poses.append(pose)
            viewer.q_camera.put(list_of_absolute_poses)
            viewer.q_pose.put(accumulated_pose)
            viewer.q_image.put(img)

        print("frame: {0:d}".format(frame_id))
        print("translation = {0:s}".format(str(pose[:3, 3])))
        print("heading={0:.1f}, pitch={1:.1f}, roll={2:.1f}".format(heading, pitch, roll))
        dist_to_board = np.sqrt(pose[:3, 3].T @ pose[:3, 3])
        distances_to_target.append(dist_to_board)
        frame_id_for_plot.append(frame_id)

        cv2.putText(frame_with_corners,'dist to board={0:f}'.format(dist_to_board),(10, 30),cv2.FONT_ITALIC, 1.0, (0, 0, 255), thickness=2)
        cv2.imshow("frame".format(frame_id), frame_with_corners)
        cv2.waitKey(33)

    import matplotlib.pyplot as plt
    plt.figure("distance to target")
    plt.plot(np.array(frame_id_for_plot), np.array(distances_to_target))
    plt.xlabel('frame_id')
    plt.ylabel('distance [m]')
    plt.grid()
    plt.show(block=True)