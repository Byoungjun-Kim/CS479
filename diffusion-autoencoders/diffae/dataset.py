"""
The codes are modified.

Link:
    - [CelebAHQ]
        - https://github.com/pytorch/vision/
          blob/677fc939b21a8893f07db4c1f90482b648b6573f/torchvision/datasets/celeba.py#L15-L189
    - [D2CCrop]
        - https://github.com/phizaz/diffae/
          blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/dataset.py#L193-L217
"""
import csv
import os
import re

import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from tqdm import tqdm
import torch
import json
import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from .utils.light_encoder import LightEncoder

class your_dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):

        # get list of folder and file names
        self.folders  = glob.glob(path+'/*/')
        self.data     = glob.glob(path+'/*/dir_*.jpg')
        self.light_encoder = LightEncoder(in_dim=3, embed_level=1, include_input=True)

        self.classes  = []
        for folder in self.folders:
            self.classes.append(folder.split('/')[-2])
        self.classes.sort()
        self.data.sort()

        self.transform = transform

        print('{:d} files {:d} classes in the dataset'.format(len(self.data), len(self.classes)))

    def __getitem__(self, index):

        fname = self.data[index]
        folder = fname.split('/')[-2]
        label = self.classes.index(folder)

        img = Image.open(fname)
        directory = os.path.dirname(fname)
        direction_id = int(fname.split('/')[-1].split('dir_')[1].split('_')[0])
        with open(directory + '/meta.json', 'r') as json_file:
            meta = json.load(json_file)
            directions = meta['directions']
            for direction in directions:
                if direction['direction_id'] == direction_id:
                    brightness_normalization = direction['brightness_normalization']
                    phi = direction['phi']
                    theta = direction['theta']
                    break

        light_source = torch.tensor([brightness_normalization, phi, theta])
        return self.transform(img), self.light_encoder.encode(light_source), label

    def __len__(self):
        return len(self.data)


def get_dataset(dataset_path, transform=None):
    """Get torchvision dataset.

    Args:
        name (str): Name of one of the following datasets,
            celeba: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
            celebahq: http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
        split (str): One of [`train`, `val`, `test`].
        transform (callable): A transform function that takes in an PIL image and returns a transformed version.

    Returns:
        dataset: A dataset class.
    """
    #transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    train_dataset = your_dataset(dataset_path, transform)

    return train_dataset


class D2CCrop:
    """
    Almost same code as
        - https://github.com/phizaz/diffae/blob/865f1926ce0d994e4a8dc2b5b250d57f519cadc1/dataset.py#L193-L217
    """
    def __init__(self):
        cx = 89
        cy = 121
        self.x1 = cy - 64
        self.x2 = cy + 64
        self.y1 = cx - 64
        self.y2 = cx + 64

    def __call__(self, img):
        img = torchvision.transforms.functional.crop(
            img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1,
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2}'


def get_torchvision_transforms(cfg, mode):
    assert mode in {'train', 'test'}
    if mode == 'train':
        transforms_cfg = cfg['train']['dataset']
    else:
        transforms_cfg = cfg['test']['dataset']

    transforms = []
    for t in transforms_cfg:
        if hasattr(torchvision.transforms, t['name']):
            transform_cls = getattr(torchvision.transforms, t['name'])(**t['params'])
        elif t['name'] == 'D2CCrop':
            # For CerebA (not CelabA-HQ) dataset, D2C Crop is applied first.
            transform_cls = D2CCrop()
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def load_image_pillow(image_path):
    with Image.open(image_path) as img:
        image = img.convert('RGB')
    return image
