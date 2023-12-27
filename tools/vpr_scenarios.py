from pathlib import Path
from typing import List, Set

import numpy as np


class VprScenario:
    entrance_start_frame: int
    entrance_good_frame: int
    entrance_end_frame: int
    exit_start_frame: int
    exit_good_frame: int
    exit_end_frame: int
    data_path: Path
    K: np.ndarray

    def get_all_frame_indices(self) -> Set[int]:
        return set(list(np.arange(self.entrance_start_frame, self.entrance_end_frame+1)) + list(np.arange(self.exit_start_frame, self.exit_end_frame+1)))

class Scenario_2023_11_27T90_37_23(VprScenario):
    entrance_start_frame = 300
    entrance_good_frame = 400
    entrance_end_frame = 450
    exit_start_frame = 8360
    exit_good_frame = 8450
    exit_end_frame = 8495
    junction_start_frame = 2095
    junction_exit_frame = 2350
    junction_2_start_frame = 2932
    junction_2_exit_frame = 3090
    data_path = Path('/october23/tunnels/DATA/2023-11-27-20231207T082758Z-001/2023-11-27T90-37-23/')
    K: np.ndarray = np.array([[1334.572266, 0, 960.029785],
                              [.0, 1334.572266, 728.388794],
                              [.0, .0, 1.0]])

# hfov is ~110[deg] (GoPro Hero10)
# resolution is 768*432
# tan(hfov_rad/2) = (W/2) / f_px -> f_px = (W/2) / tan(hfov_rad/2)
# np.tan(110 / 180 * np.pi / 2) = 1.4281480067421144
# f_px = 384 / 1.4281480067421144 ~ 268
# K = [268 0 384
#      0 268 216
#      0   0   1]
class GoProScenario(VprScenario):
    entrance_start_frame = 0
    entrance_good_frame = 500
    entrance_end_frame = 1549
    data_path = Path('/october23/tunnels/DATA/BLUR')
    K = np.array([[268.0, .0, 384.0],
                  [.0, 268.0, 216.0],
                  [.0,    .0,   1.0]])