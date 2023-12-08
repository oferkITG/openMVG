""" encode_db_w_netvlad
This script encodes a given dataset with netvlad
"""
import argparse
import torch
from utils.vpr_utils import netvlad_transform
from loaders.VPRDataset import VPRDataset
from models.NetVLAD import NetVLAD
import os
from os.path import join

# example command: data/assets/vpr/dataset1/Shelef_AAA_0504_511/  weights/pretrained_vgg16_pitts30k_netvlad_from_matlab.pth output/

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("encoder_path", help="a .pth file to netvlad encoder")
    arg_parser.add_argument("output_path", help="output path")
    arg_parser.add_argument("--img_size",
                            help="resizing size for image embedding")
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")

    args = arg_parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    dataset_id = os.path.basename(os.path.normpath(args.dataset_path))
    if torch.cuda.is_available():
        device_id = args.device
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    print("Create and load the model for image encoding and retrieval")
    net = NetVLAD()
    net.load_state_dict(torch.load(args.encoder_path, map_location=device_id))

    print("Create the dataset and the loader")
    img_size = args.img_size
    if img_size is not None: # if the size is None we will do dynamic resizing
        img_size = int(img_size)
        batch_size = 4
    else:
        batch_size = 1
    img_size = 2000
    netvlad_transform = netvlad_transform(img_size)
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4}
    dbset = VPRDataset(args.dataset_path, transform=netvlad_transform)
    dbloader = torch.utils.data.DataLoader(dbset, **loader_params)

    db_global_descriptors = None
    img_paths = []
    i = 0
    print("Starting to encode images at {}".format(args.dataset_path))
    with torch.no_grad():
        net.eval()
        net.to(device)
        for minibatch in dbloader:
            imgs = minibatch.get('img').to(device)
            global_desc = net(imgs)
            if db_global_descriptors is None:  #
                m = len(dbloader.dataset)
                n = global_desc.shape[1]
                db_global_descriptors = torch.zeros((m, n), device=device)
            d = imgs.shape[0]
            db_global_descriptors[i:(i + d), :] = global_desc
            i = i + d
    print("encoding of {} images completed".format(len(dbloader.dataset)))
    output_name = join(args.output_path,"{}_netvlad_encoding.pth".format(dataset_id))
    torch.save({"encoding":db_global_descriptors,
                "img_files":dbloader.dataset.img_files,
                "img_ids":dbloader.dataset.img_ids,
                "metadata":dbloader.dataset.metadata,
                "dataset_path":dbloader.dataset.dataset_path}, output_name)
    print("results saved to {}".format(join(args.output_path,output_name)))

