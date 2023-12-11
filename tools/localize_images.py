import os
from typing import Type, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from tools.vpr.encode_dataset import VprEncoder
from tools.vpr.options import options
from tools.vpr.utils.vpr_utils import get_nearest_neighbors
from tools.vpr_scenarios import Scenario_2023_11_27T90_37_23, VprScenario

device: torch.device = torch.device('cpu')
encoder = VprEncoder(encoder_type='cosplace',
                     encoder_options=options['cosplace'],
                     device_id=device)


def search_db(db: torch.Tensor, db_offset: int, target_encoding: torch.Tensor, label: str, results_path: Path):
    search_db = db[db_offset:]
    similarity = torch.matmul(search_db, torch.unsqueeze(target_encoding, 0).transpose(0, 1)).squeeze() / ((torch.norm(search_db, dim=1)) * torch.norm(target_encoding))
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

if __name__ == "__main__":
    # read encodings
    scenario: Type[VprScenario] = Scenario_2023_11_27T90_37_23()

    data_path: Path = scenario.data_path
    encodings_path: Path = data_path / 'encodings' / 'cosplace_small_res'
    images_path: Path = data_path / 'images'
    results_path: Path = data_path / 'results'
    os.makedirs(results_path, exist_ok=True)

    db_vec: torch.Tensor = torch.load(encodings_path / 'encodings.torch')
    TARGET_FRAME: int = scenario.junction_2_start_frame
    target_img: np.ndarray = cv2.imread(str(images_path / "frame{0:d}.jpg".format(TARGET_FRAME)))
    cv2.imshow("target frame ({0:d})".format(TARGET_FRAME), target_img)

    target_encoding: torch.Tensor = db_vec[TARGET_FRAME].to(device)
    search_db(db=db_vec, db_offset=0, target_encoding=target_encoding, label='all_frames', results_path=results_path)
    db_offset: int = scenario.junction_2_exit_frame
    search_db(db=db_vec, db_offset=db_offset, target_encoding=target_encoding, label='after_{0:d}'.format(db_offset), results_path=results_path)

    plt.show()
    cv2.waitKey(0)


