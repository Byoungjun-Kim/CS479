from PIL import Image
import matplotlib.pyplot as plt

from diffae import DiffusionAutoEncodersInterface
from diffae.utils.light_encoder import LightEncoder
import torch
import glob
import random
import json
import re
import torch.nn.functional as F
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', type=str, help='dataset path fortrain', default='')  # should be illumination dataset path
parser.add_argument('-de', '--device', type=int, help='device', default=0)
parser.add_argument('-c', '--model_ckpt', type=str, help='checkpoint name', default='')
parser.add_argument('-o', '--output', type=str, help='saved checkpoint directory', default=None)
args = parser.parse_args()

args = vars(args)
device = f'cuda:{args["device"]}'
light_encoder = LightEncoder(in_dim=3, embed_level=1, include_input=True)
mode = 'infer'
diffae = DiffusionAutoEncodersInterface(args, mode)

folders  = glob.glob(args['dataset_path'] + '/test/*/')

random_integers = random.sample(range(len(folders)), 5)
total_mse_loss_sum = 0
ground_mse_loss_sum = 0
num_count = 0
for i in random_integers:
    folder = folders[i]
    data_list = glob.glob(folder + 'dir_*.jpg')
    with open(folder + '/meta.json', 'r') as json_file:
        meta = json.load(json_file)
        directions = meta['directions']
        direction_dict = {}
        for direction in directions:
            direction_dict[direction['direction_id']] = {
                'brightness': direction['brightness_normalization'],
                'phi': direction['phi'],
                'theta': direction['theta']
            }

    query_data_list = data_list[:24]
    test_data_list = data_list[24:]
    
    for query_data in query_data_list:
        query_img = Image.open(query_data)

        for test_data in test_data_list:
            direction_id = int(re.search(r'dir_(\d+)_mip', test_data).group(1))
            to_dir = light_encoder.encode(torch.tensor(list(direction_dict[direction_id].values())))
            result = diffae.infer_manipulate(query_img, to_dir)
            output = result['output']
            test_image = Image.open(test_data)
            trans = diffae.transforms(test_image)
            trans = diffae.unnormalize(trans).permute(1, 2, 0).cpu().detach().numpy()

            trans_query = diffae.transforms(query_img)
            trans_query = diffae.unnormalize(trans_query).permute(1, 2, 0).cpu().detach().numpy()

            ground_tensor = torch.tensor(np.array(trans).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
            query_tensor = torch.tensor(np.array(trans_query).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
            output_tensor = torch.tensor(np.array(output).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
            mse_loss = ((ground_tensor - output_tensor)**2).sum().sqrt()
            ground_mse_loss = ((ground_tensor - query_tensor)**2).sum().sqrt()
            total_mse_loss_sum += mse_loss
            ground_mse_loss_sum += ground_mse_loss
            num_count += 1

print(f'loss is: {total_mse_loss_sum / num_count}')
print(f'Ground loss is: {ground_mse_loss_sum / num_count}')
