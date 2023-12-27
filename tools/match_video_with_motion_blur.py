import glob
import os
from typing import List
import cv2
import numpy as np
import torch
from tools.features.base import KeypointData, KeypointMatchingResults
from tools.features.outlier_filter import FilterType, outlier_filtering, keep_inliers
from tools.features.superglue import SuperGlue
from tools.features.superpoint import SuperPoint
from tools.localize_images_with_viz import OutlierFilteringResult, apply_outlier_filtering, KeepInliersResult

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_frame_index(image_path: str) -> int:
    return int(os.path.split(image_path)[1].split('.jpg')[0].split('frame')[1])

if __name__ == "__main__":
    list_of_image_paths: List[str] = glob.glob('/october23/tunnels/DATA/BLUR/images/*.jpg')
    list_of_image_paths = sorted(list_of_image_paths, key=lambda x: get_frame_index(x))
    prev_keypoint_data: KeypointData = None
    prev_image: np.ndarray = None
    superglue = SuperGlue({'weights': 'outdoor'}).to(device).eval()
    superpoint = SuperPoint({}).to(device).eval()

    for img_path in list_of_image_paths:
        curr_img: np.ndarray = cv2.imread(img_path)
        curr_img_gray: np.ndarray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("current img", curr_img)
        cv2.waitKey(int(1000*1/60))

        curr_keypoint_data: KeypointData = superpoint.detectAndCompute(curr_img_gray, device)

        if prev_keypoint_data is not None:
            kp_matching_res: KeypointMatchingResults = superglue.match(curr_keypoint_data, prev_keypoint_data)

            kpts_train = np.array([kp_matching_res.train_keypoints[m.trainIdx].pt for m in kp_matching_res.matches])
            kpts_query = np.array([kp_matching_res.query_keypoints[m.queryIdx].pt for m in kp_matching_res.matches])
            mat, inliers = outlier_filtering(FilterType.FUNDAMENTAL, kpts_train, kpts_query, ransac_inlier_thr=3, K=np.eye(3))
            inlier_matches_idx = np.where(inliers)[0].astype(np.int32).tolist()
            inlier_matches = [kp_matching_res.matches[i] for i in inlier_matches_idx]
            inliers: KeepInliersResult = keep_inliers(inlier_matches, kp_matching_res.query_keypoints,
                                                      kp_matching_res.train_keypoints)

            match_image: np.ndarray = cv2.drawMatches(curr_img,
                                                      inliers.kpts_query,
                                                      prev_image,
                                                      inliers.kpts_train,
                                                      inliers.matches,
                                                      None)
            cv2.imshow("matches", match_image)

        prev_keypoint_data = curr_keypoint_data
        prev_image = curr_img
