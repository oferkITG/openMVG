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

    def get_all_frame_indices(self) -> Set[int]:
        return set(list(np.arange(self.entrance_start_frame, self.entrance_end_frame+1)) + list(np.arange(self.exit_start_frame, self.exit_end_frame+1)))

class Scenario_2023_11_27T90_37_23(VprScenario):
    entrance_start_frame = 300
    entrance_good_frame = 400
    entrance_end_frame = 450
    exit_start_frame = 8360
    exit_good_frame = 8450
    exit_end_frame = 8495
    data_path = Path('/home/gabi/Work/october23/tunnels/DATA/2023-11-27-20231207T082758Z-001/2023-11-27T90-37-23/')