import glob
import os
from typing import List
import cv2
import numpy as np
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, COLOR_BGR2GRAY, imread, cvtColor, findChessboardCorners, cornerSubPix, calibrateCamera, getOptimalNewCameraMatrix, undistort, imshow
from scipy.spatial.transform import Rotation

from tools.pangolin_viz import Viewer


def resize(img: np.ndarray, factor: int = 1):
    if factor == 1:
        return img
    else:
        return cv2.resize(img, [img.shape[1]//factor, img.shape[0]//factor])
def get_pose(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    rotation_matrix: np.ndarray = cv2.Rodrigues(rot)[0]
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = t.squeeze()
    return np.linalg.inv(T)

def get_frame_id(name: str) -> int:
    return int(name.split('.')[0].split('frame')[1])

if __name__ == "__main__":
    """
    This test performs camera intrinsics, extrinsics and distortion, using OpenCV's calibrateCamera()
    which is based on Zhang's method
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    @return:
    """
    # termination criteria
    IMAGE_DOWNSIZE_FACTOR: int = 3
    VISUALIZATION: bool = False
    DISPLAY_IMAGE_ID: int = 0
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    SKIP_FRAMES: int = 50
    N_FRAMES: int = 20
    N_ROWS: int = 8
    N_COLS: int = 6
    PATTERN_SIZE = (N_ROWS, N_COLS)
    SQUARE_SIZE: float = 2.714*1e-2
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    WORLD_POINTS = np.zeros((N_COLS * N_ROWS, 3), np.float32)
    WORLD_POINTS[:, :2] = SQUARE_SIZE * np.mgrid[0:N_ROWS, 0:N_COLS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images_path: str = "/october23/tunnels/K9/calib/GX010898/images"
    images = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
    list_of_frames: List[np.ndarray] = []
    list_of_frame_indices: List[int] = []
    show_image_flag: bool = False
    for idx, image_path in enumerate(images):
        if idx % SKIP_FRAMES != 0:
            continue
        frame_id: int = get_frame_id(image_path)

        img = resize(cv2.imread(image_path), factor=IMAGE_DOWNSIZE_FACTOR)
        gray = cvtColor(img, COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = findChessboardCorners(image=gray,
                                             patternSize=PATTERN_SIZE,
                                             corners=None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(WORLD_POINTS)
            corners2 = cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            list_of_frames.append(img)
            list_of_frame_indices.append(frame_id)

    if VISUALIZATION:
        viewer: Viewer() = Viewer()
        cameras: List = []
        camera_pose = np.eye(4)
        cameras.append(camera_pose)

    ret, K, distCoeffs, rvecs, tvecs = calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("K = {0:s}".format(str(K)))
    print("distortion = {0:s}".format(str(distCoeffs)))
    if VISUALIZATION:
        accumulated_pose = np.eye(4)
        list_of_absolute_poses = [accumulated_pose]
        viewer.q_camera.put(list_of_absolute_poses)
        viewer.q_pose.put(accumulated_pose)

    list_of_frames = list_of_frames[:N_FRAMES]
    rvecs = rvecs[:N_FRAMES]
    tvecs = tvecs[:N_FRAMES]
    list_of_frame_indices = list_of_frame_indices[:N_FRAMES]
    for (frame_id, rot, translation, img ) in zip(list_of_frame_indices, rvecs, tvecs, list_of_frames):
        pose = get_pose(rot, translation)
        if VISUALIZATION:
            list_of_absolute_poses.append(pose)
            viewer.q_camera.put(list_of_absolute_poses)
            viewer.q_pose.put(accumulated_pose)
            viewer.q_image.put(img)

        projection_image: np.ndarray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        projection, jacobian = cv2.projectPoints(objpoints[DISPLAY_IMAGE_ID], rot, translation, K, distCoeffs)
        for proj in projection:
            pixel = (int(proj.squeeze()[0]), int(proj.squeeze()[1]))
            cv2.circle(projection_image, pixel, 10, (0, 255, 0), -1)
        # projection_image = cv2.resize(projection_image, [projection_image.shape[1]//3, projection_image.shape[0]//3])
        imshow('expected projection', projection_image)

        # print("pose = {0:s}".format(str(pose)))
        position_world: np.ndarray = pose[:3, 3]
        rotation_world: np.ndarray = pose[:3, :3]
        heading, pitch, roll = Rotation.from_matrix(rotation_world).as_euler('ZYX', degrees=True)
        print("frame: {0:d}".format(frame_id))
        print("translation = {0:s}".format(str(position_world)))
        print("heading={0:.1f}, pitch={1:.1f}, roll={2:.1f}".format(heading, pitch, roll))

        cv2.waitKey(0)


