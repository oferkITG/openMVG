import sys

import cv2

sys.path.append("../")
from skimage.io import imread
from torch.utils.data import Dataset
from os.path import join
import os
import yaml
from tools.vpr.utils.vpr_utils import netvlad_transform
import rasterio
import numpy as np
import pickle

def is_img(f):
    suffixes = [".jpg", ".JPG", ".PNG", ".png", ".tiff", "tif"]
    for s in suffixes:
        if f.endswith(s):
            return True
    return False

class VPRDataset(Dataset):
    """
        A class used to represent a pytorch Dataset for visual place recognition

        Attributes
        ----------
        dataset_path : str
            a path to the dataset location with images
        transform : transform object (torchvision.transforms)
            a transformation (or composition of transformations) to be applied on an image when loaded
    """

    def __init__(self, dataset_path, transform=None):
        """
        :param dataset_path: (str) a path to the physical location of the images
        :return: an instance of VPRDataset
        """
        super(VPRDataset, self).__init__()

        self.dataset_path = dataset_path

        if os.path.isfile(dataset_path):

            self.img_files = [dataset_path]
            self.img_ids = [f.split(".")[0] for f in self.img_files]
        else:
            self.img_files = [f for f in list(os.listdir(dataset_path)) if is_img(f)]
            self.img_files.sort()
            self.img_ids = [f.split(".")[0] for f in self.img_files]
        # collect meta-data, if such exists, for evaluation purposes
        self.metadata = None
        landmarks_meta_file = join(dataset_path, "annot_landmarks.yaml")
        tiles_meta_file = join(dataset_path, "annot_tiles.yaml")
        transformations_pkl_file = join(dataset_path, "view_transformations.pkl")
        if os.path.exists(landmarks_meta_file):
            with open(landmarks_meta_file, 'r') as f:
                self.metadata = yaml.safe_load(f)
        elif os.path.exists(landmarks_meta_file):
            with open(tiles_meta_file, 'r') as f:
                self.metadata = yaml.safe_load(f)
        elif os.path.exists(transformations_pkl_file):
            with open(transformations_pkl_file, 'rb') as f:
                self.metadata = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def plot(self):
        # deprecated
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        img_path = join(self.dataset_path, self.img_files[0])
        img = rasterio.open(img_path).read()
        img_arr = np.array(img).transpose(1, 2, 0)
        print(img_arr.shape)
        img_arr = netvlad_transform(2000)(img_arr).numpy().transpose(1, 2, 0).astype(int)
        print(np.max(img_arr))
        print(np.min(img_arr))
        plt.imshow(img_arr)
        plt.show()

    def __getitem__(self, idx):
        img_path = join(self.dataset_path,self.img_files[idx])
        is_tile = True
        if img_path.endswith("tif") or img_path.endswith("tiff"):
            img = rasterio.open(img_path).read()
            img_arr = np.array(img).transpose(1, 2, 0)
            img = img_arr
        else:
            img = imread(img_path)
            if img.shape[2] == 4:
                # slice off the alpha channel
                img = img[:, :, :3]
            is_tile = False
        if self.transform is not None:
            if is_tile:
                img = self.transform["orthophoto_transform"](img)
            else:
                img = self.transform["shelef_transform"](img)
        sample = {'img': img}
        return sample


class FromStorageVPRDataset(Dataset):

    def __init__(self, image_paths_dict, transform=None):
        self.image_paths_dict = image_paths_dict
        self.idx2name = {i: name for i, name in enumerate(self.image_paths_dict.keys())}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths_dict)

    def __getitem__(self, idx):
        name = self.idx2name[idx]
        img_path = self.image_paths_dict[name]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform["shelef_transform"](img)
        return {'name': name,
                'img': img}



class InMemoryVPRDataset(Dataset):

    def __init__(self, images_dict, transform=None):
        self.images_dict = images_dict
        self.idx2name = {i: name for i, name in enumerate(self.images_dict.keys())}
        self.transform = transform

    def __len__(self):
        return len(self.images_dict)


    def __getitem__(self, idx):
        name = self.idx2name[idx]
        img = cv2.cvtColor(self.images_dict[name], cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform["shelef_transform"](img)
        return {'name': name,
                'img': img}


