import os
import pickle
import argparse
import cv2
import torch
from loaders.VPRDataset import is_img
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from utils.vpr_utils import base_transform
import matplotlib
matplotlib.use('TKAgg')

def show_all_views(first_match, num_views):
    props = list(first_match.values())[0]
    nn_path = props["file"]
    view_id = nn_path.split("_")[-1]
    view_suffix = view_id.split(".")[-1]
    for i in range(num_views):
        view_path = nn_path.replace(view_id, "{}.{}".format(i, view_suffix))
        print(view_path)
        if os.path.exists(view_path):
            print("view exists - plotting view")
            view_img = cv2.cvtColor(cv2.imread(view_path), cv2.COLOR_BGR2RGB)
            plt.title("view {}".format(i))
            plt.imshow(view_img)
            plt.tight_layout()
            plt.axis('off')
            plt.show()



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("results_file", help="path to the localization results file")
    arg_parser.add_argument("--device", default="cuda:0", help="device identifier")

    args = arg_parser.parse_args()
    input_dataset = args.dataset_path
    results_file = args.results_file
    print("Loading localization results file {}".format(results_file))
    with open(results_file, 'rb') as f:
        res = pickle.load(f)

    if torch.cuda.is_available():
        device_id = args.device
    else:
        device_id = 'cpu'
    device = torch.device(device_id)


    img_files = os.listdir(input_dataset)
    img_ids = [f.split(".")[0] for f in img_files if is_img(f)]
    img_ids.sort()
    img_files.sort()

    x = 0
    for q_idx, query_id in enumerate(img_ids):
        #if query_id != ("AAE_0499"):
        #    continue
        loc = res[query_id]

        #show_all_views(loc[0], 10)
        #break

        shelef_filepath = os.path.join(input_dataset, img_files[q_idx])
        print("Evaluating matches for shelef img {}".format(query_id))
        print("Ploting image: {}".format(img_files[q_idx]))

        # Read image and plot
        shelef_img = cv2.cvtColor(cv2.imread(shelef_filepath), cv2.COLOR_BGR2RGB)

        plt.title(query_id)
        plt.imshow(shelef_img)
        plt.tight_layout()
        plt.axis('off')
        plt.show()

        scale_percent = 25 # percent of original size
        width = int(shelef_img.shape[1] * scale_percent / 100)
        height = int(shelef_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        shelef_resized = cv2.resize(shelef_img, dim, interpolation=cv2.INTER_AREA)

        for i, match in loc.items():
            print("Neighbor {}".format(i+1))
            for nn_id, props in match.items():
                print("Id: {}".format(nn_id))
                for k, v in props.items():
                    print("{}:{}".format(k, v))
                    if k == "file":
                        # treat tiff
                        if v.endswith(".tiff") or v.endswith(".tif"):
                            img = rasterio.open(v).read()
                            img_arr = np.array(img).transpose(1, 2, 0)
                            ortho_view_img = base_transform(2000)(img_arr).numpy().transpose(1, 2, 0).astype(int)
                        else:
                            ortho_view_img = cv2.cvtColor(cv2.imread(v), cv2.COLOR_BGR2RGB)
                            scale_percent = 25  # percent of original size
                            width = int(ortho_view_img.shape[1] * scale_percent / 100)
                            height = int(ortho_view_img.shape[0] * scale_percent / 100)
                            dim = (width, height)
                            # resize image
                            ortho_resized = cv2.resize(ortho_view_img, dim, interpolation=cv2.INTER_AREA)


                        plt.title(nn_id)
                        plt.imshow(ortho_view_img)
                        plt.tight_layout()
                        plt.axis('off')
                        plt.show()




