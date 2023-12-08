import argparse
from pathlib import Path

import torch
from tools.vpr.loaders.VPRDataset import VPRDataset, InMemoryVPRDataset, FromStorageVPRDataset
import os
from os.path import join
from tools.vpr.models.NetVLAD import NetVLAD
from tools.vpr.models.cosplace.cosplace_network import GeoLocalizationNet
from tools.vpr.options import options
'''
example args:
data/assets/vpr/dataset1/ output/ --encoder netvlad
data/assets/vpr/dataset1/ output/ --encoder cosplace
data/assets/vpr/dataset1/ output/ --encoder eigenplaces
/mnt/artifacts/assets/TrainSetV2/Aza output/ --encoder eigenplaces
/mnt/rawdata/aza/ortophoto/tiff/ output/ --encoder eigenplaces --single_folder
output/vpr_nfov_b1 output/vpr_nfov_b1 --encoder eigenplaces
output/vpr_b1_small/ output/vpr_b1_small/ --encoder eigenplaces
'''

class VprEncoder:
    def __init__(self, encoder_type, encoder_options, device_id):
        self.device_id = device_id
        self.encoder_type = encoder_type
        self.encoder_options = encoder_options
        if torch.cuda.is_available():
            device_id = device_id
        else:
            device_id = 'cpu'
        self.device = torch.device(device_id)

        print("Encoding a dataset with {}".format(encoder_type))
        print("Create and load the model for image encoding and retrieval")
        if encoder_type in ("cosplace", "cosplace_small_res"):
            # https://github.com/gmberton/CosPlace
            self.backbone = encoder_options["backbone"]
            self.dim = encoder_options["dim"]
            self.transform = encoder_options["transforms"]
            # self.net = torch.hub.load("gmberton/cosplace", "get_trained_model",
            #                      backbone=self.backbone, fc_output_dim=self.dim)
            self.net = GeoLocalizationNet(self.backbone, fc_output_dim=self.dim)
            self.net.load_state_dict(torch.load(Path(__file__).parent.parent.parent.joinpath('tools', 'vpr', 'assets', 'ResNet50_2048_cosplace.pth'),
                                                map_location=device_id))
            self.net = self.net.eval()


        elif encoder_type == "netvlad":
            self.net = NetVLAD()
            self.net.load_state_dict(torch.load(encoder_options["weights"], map_location=device_id))
            self.transform = encoder_options["transforms"]
            self.dim = encoder_options["dim"]
        elif encoder_type == "eigenplaces":
            # https://github.com/gmberton/EigenPlaces
            self.backbone = encoder_options["backbone"]
            self.dim = encoder_options["dim"]
            self.transform = encoder_options["transforms"]
            self.net = torch.hub.load("gmberton/eigenplaces",
                                 "get_trained_model",
                                 backbone=self.backbone, fc_output_dim=self.dim)
        else:
            raise NotImplementedError
        with torch.no_grad():
            self.net.eval()
            self.net.to(self.device)

    def encode_images(self, image_paths_dict, images_dict=None, verbose: bool = False, batch_size: int = 1):
        encoding_dict = dict()
        loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0}
        if images_dict:
            dataset = InMemoryVPRDataset(images_dict, transform=self.transform)
        elif image_paths_dict:
            dataset = FromStorageVPRDataset(image_paths_dict, transform=self.transform)
        else:
            raise NotImplementedError("Must provide images_dict or image_path_dict")
        loader = torch.utils.data.DataLoader(dataset, **loader_params)
        # print("Create the dataset and the loader for dataset {}".format(dataset_id))

        # db_global_descriptors = torch.zeros((len(dataset), self.dim)).to(self.device)
        with torch.no_grad():
            for i, minibatch in enumerate(loader):
                sample_name = minibatch['name']
                if verbose:
                    print("Starting to encode images at {}".format(sample_name))
                imgs = minibatch['img'].to(self.device)
                global_desc = self.net(imgs)
                d = imgs.shape[0]
                # db_global_descriptors[i:(i + d), :] = global_desc
                i = i + d
                encoding_dict[sample_name] = global_desc
        if verbose:
            print("encoding of {} images completed".format(len(loader.dataset)))
        return encoding_dict






def encode_dataset(dataset_path, output_path,
                   encoder_type, encoder_options,
                   single_folder, device_id):
    os.makedirs(output_path, exist_ok=True)
    if torch.cuda.is_available():
        device_id = device_id
    else:
        device_id = 'cpu'
    device = torch.device(device_id)

    print("Encoding a dataset with {}".format(encoder_type))
    print("Create and load the model for image encoding and retrieval")
    if encoder_type in ("cosplace", "cosplace_reduce_shelef_x4"):
        # https://github.com/gmberton/CosPlace
        backbone = encoder_options["backbone"]
        dim = encoder_options["dim"]
        transform = encoder_options["transforms"]
        net = torch.hub.load("gmberton/cosplace", "get_trained_model",
                             backbone=backbone, fc_output_dim=dim)
    elif encoder_type == "netvlad":
        net = NetVLAD()
        net.load_state_dict(torch.load(encoder_options["weights"], map_location=device_id))
        transform = encoder_options["transforms"]
        dim = encoder_options["dim"]
    elif encoder_type == "eigenplaces":
        # https://github.com/gmberton/EigenPlaces
        backbone = encoder_options["backbone"]
        dim = encoder_options["dim"]
        transform = encoder_options["transforms"]
        net = torch.hub.load("gmberton/eigenplaces",
                             "get_trained_model",
                             backbone=backbone, fc_output_dim=dim)
    else:
        raise NotImplementedError


    if single_folder:
        dataset_ids = [os.path.basename(os.path.normpath(dataset_path))]
    else:
        dataset_ids = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    save_in_folder = False
    if dataset_path == output_path:
        save_in_folder = True
        output_path = output_path
    else:
        root_id = os.path.basename(os.path.normpath(dataset_path))
        output_path = os.path.join(output_path, root_id)
        os.makedirs(output_path, exist_ok=True)

    for dataset_id in dataset_ids:
        if single_folder:
            curr_dataset_path = os.path.join(dataset_path)
        else:
            curr_dataset_path = os.path.join(dataset_path, dataset_id)
        print("Create the dataset and the loader for dataset {}".format(dataset_id))
        batch_size = 1
        loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 4}
        dataset = VPRDataset(curr_dataset_path, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, **loader_params)
        db_global_descriptors = torch.zeros((len(dataset), dim)).to(device)
        img_paths = []
        i = 0
        print("Starting to encode images at {}".format(dataset_id))
        with torch.no_grad():
            net.eval()
            net.to(device)
            for i,minibatch in enumerate(loader):
                print('Encode image: %s'%i)
                imgs = minibatch.get('img').to(device)
                global_desc = net(imgs)
                d = imgs.shape[0]
                db_global_descriptors[i:(i + d), :] = global_desc
                i = i + d
        print("encoding of {} images completed".format(len(loader.dataset)))
        if save_in_folder:
            output_name = join(join(output_path, dataset_id), "{}_{}_encoding.pth".format(dataset_id, encoder_type))
        else:
            output_name = join(output_path, "{}_{}_encoding.pth".format(root_id + "_" + dataset_id, encoder_type))
        torch.save({"encoding": db_global_descriptors,
                    "img_files": loader.dataset.img_files,
                    "img_ids": loader.dataset.img_ids,
                    "metadata": loader.dataset.metadata,
                    "dataset_path": loader.dataset.dataset_path}, output_name)
        print("results saved to {}".format(join(output_path, output_name)))


if __name__ == "__main__":

    stop
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("output_path", help="output path")
    arg_parser.add_argument("--device", default="cuda:0",
                            help="device identifier")
    arg_parser.add_argument("--encoder", default="netvlad",
                            help="supported models: netvlad, cosplace, eigenplaces")
    arg_parser.add_argument("--single_folder", action='store_true')

    args = arg_parser.parse_args()
    encoder_options = options[args.encoder]
    encode_dataset(args.dataset_path, args.output_path,
                       args.encoder, encoder_options,
                       args.single_folder, args.device)

