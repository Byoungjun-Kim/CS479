#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import math
import pdb
import json
import glob
import os
from PIL import Image
import torchvision.transforms as transforms

class your_dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):

        # get list of folder and file names
        self.folders  = glob.glob(path+'/*/')
        self.data     = glob.glob(path+'/*/dir_*.jpg')

        # make the list of class names
        self.data.sort()

        self.transform = transform

        print('{:d} files in the dataset'.format(len(self.data)))

    def __getitem__(self, index):

        fname = self.data[index]

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
        return self.transform(img), light_source

    def __len__(self):
        return len(self.data)

def get_data_loader(batch_size):
    train_path = '/root/data/train'
    test_path = '/root/data/validate'
    transform=transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    train_dataset = your_dataset(train_path, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=10,
        drop_last=False,
    )

    test_dataset = your_dataset(test_path, transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5,
        num_workers=10,
        drop_last=False,
    )

    return train_loader,test_loader
