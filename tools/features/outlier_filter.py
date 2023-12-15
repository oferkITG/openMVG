import dataclasses
import enum
import pickle
from typing import List, Optional, Dict

import numpy as np

from tools.features.base import KeypointMatchingResults, KeypointData
import cv2


class FilterType(enum.Enum):
    NONE = -1
    EPIPOLAR = 0
    HOMOGRAPHY = 1
    FUNDAMENTAL = 2


def outlier_filtering(filter_matches_type: FilterType, kpts0, kpts1, ransac_inlier_thr: float, K: np.ndarray = None):
    if filter_matches_type is FilterType.NONE:
        inliers = np.ones(kpts0.shape[0])
        mat = None
    elif filter_matches_type is FilterType.EPIPOLAR:
        mat, inliers = cv2.findEssentialMat(kpts0, kpts1, K, cv2.RANSAC, 0.999, 3)
    elif filter_matches_type is FilterType.FUNDAMENTAL:
        mat, inliers = cv2.findFundamentalMat(kpts0, kpts1, cv2.RANSAC, ransac_inlier_thr, 0.999, 100000)
    elif filter_matches_type is FilterType.HOMOGRAPHY:
        if kpts0.shape[0] < 4:
            print("cannot find homography with less than 4 points")
            inliers = np.ones(kpts0.shape[0])
            mat = None
        else:
            mat, inliers = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, ransac_inlier_thr)
    return mat, inliers

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
        return {"queryIdx":[m.queryIdx for m in self.matches],
                "trainIdx":[m.trainIdx for m in self.matches],
                "distance":[m.distance for m in self.matches]}

    def get_kpts_as_dict(self, kpts: List[cv2.KeyPoint]) -> Dict:
        return {"pt_x":[kpt.pt[0] for kpt in kpts],
                "pt_y": [kpt.pt[1] for kpt in kpts],
                "size":[kpt.size for kpt in kpts],
                "angle":[kpt.angle for kpt in kpts],
                "response":[kpt.response for kpt in kpts],
                "octave":[kpt.octave for kpt in kpts],
                "class_id":[kpt.class_id for kpt in kpts]}

    @classmethod
    def from_dict(cls, data: Dict):
        return KeepInliersResult(matches=cls.parse_matches_dict(data['matches']),
                                 kpts_query=cls.parse_kepoints_dict(data['kpts_query']),
                                 kpts_train=cls.parse_kepoints_dict(data['kpts_train']))

    def as_dict(self) -> Dict:
        return {"matches":self.get_matches_as_dict(),
                "kpts_query":self.get_kpts_as_dict(self.kpts_query),
                "kpts_train":self.get_kpts_as_dict(self.kpts_train)}

def keep_inliers(inlier_matches: List[cv2.DMatch], kpts_query: List[cv2.KeyPoint], kpts_train: List[cv2.KeyPoint]) -> KeepInliersResult:
    out_matches: List[cv2.DMatch] = []
    out_kpts_query: List[cv2.KeyPoint] = []
    out_kpts_train: List[cv2.KeyPoint] = []
    for idx, m in enumerate(inlier_matches):
        match: cv2.DMatch = cv2.DMatch()
        match.queryIdx = idx
        match.trainIdx = idx
        out_matches.append(match)
        out_kpts_query.append(kpts_query[m.queryIdx])
        out_kpts_train.append(kpts_train[m.trainIdx])
    return KeepInliersResult(matches=out_matches, kpts_query=out_kpts_query, kpts_train=out_kpts_train)

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

def apply_outlier_filtering(kp_matching_res: KeypointMatchingResults, kp_query: KeypointData, kp_train: KeypointData, query_image: np.ndarray = None, train_image: np.ndarray = None,
                            filter_type: FilterType = FilterType.HOMOGRAPHY, K: np.ndarray = None) -> OutlierFilteringResult:
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
