import glob
import os
from typing import List, Type
import torch
from pathlib import Path
from tools.vpr.encode_dataset import VprEncoder
from tools.vpr.options import options
from tools.vpr_scenarios import VprScenario, Scenario_2023_11_27T90_37_23

device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def get_frame_number(img_path: str) -> int:
    img_name: str = os.path.split(img_path)[1].split('.jpg')[0]
    return int(img_name.split('frame')[1])

if __name__ == "__main__":

    encoder_type: str = 'cosplace_small_res'
    encoder = VprEncoder(encoder_type=encoder_type,
                         encoder_options=options[encoder_type],
                         device_id=device)
    scenario: Type[VprScenario] = Scenario_2023_11_27T90_37_23()
    # valid_frames: Set[int] = scenario.get_all_frame_indices()
    data_path: Path = Path('/home/gabi/Work/october23/tunnels/DATA/2023-11-27-20231207T082758Z-001/2023-11-27T90-37-23')
    images_path: Path = scenario.data_path / 'images'
    encodings_path: Path = scenario.data_path / 'encodings' / encoder_type
    os.makedirs(encodings_path, exist_ok=True)

    list_of_image_paths: List[str] = glob.glob(os.path.join(str(images_path.resolve()), '*.jpg'))
    list_of_image_paths = sorted(list_of_image_paths, key=lambda x: get_frame_number(x))

    encodings: torch.Tensor = torch.empty((len(list_of_image_paths), 2048))

    image_paths_dict = {}
    for img_path in list_of_image_paths:
        image_paths_dict[get_frame_number(img_path)] = img_path

    encodings = encoder.encode_images(image_paths_dict=image_paths_dict, batch_size=10)
    out: torch.Tensor = torch.empty((len(list_of_image_paths), 2048))
    for frames, enc in encodings.items():
        for frame, enc in zip(frames.tolist(), enc):
            out[frame] = enc.cpu()
    output_filename: Path = encodings_path / 'encodings.torch'
    torch.save(out, output_filename)
    print("saved encodings to {0:s}".format(str(output_filename)))