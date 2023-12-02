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
        # Normalize phi and theta to be between 0 and 1
        normalized_phi = (phi + 3.141593) / (2 * 3.141593)  # Normalizing phi
        normalized_theta = 2 * theta / 3.141593  # Normalizing theta

        light_source = torch.tensor([brightness_normalization, normalized_phi, normalized_theta])
        return self.transform(img), light_source
    def __len__(self):
        return len(self.data)

def get_data_loader(batch_size, shuffle):
    train_path = 'data/train/'
    transform=transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    train_dataset = your_dataset(train_path, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=10,
        drop_last=False,
        shuffle=shuffle,
    )
    
    return train_loader
