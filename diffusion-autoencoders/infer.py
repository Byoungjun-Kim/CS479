from PIL import Image
import matplotlib.pyplot as plt

from diffae import DiffusionAutoEncodersInterface
from diffae.utils.light_encoder import LightEncoder
import torch

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, help='dataset path fortrain', default='')  # should be illumination dataset path
    parser.add_argument('-de', '--device', type=int, help='device', default=0)
    parser.add_argument('-c', '--model_ckpt', type=str, help='checkpoint name', default='')
    parser.add_argument('-o', '--output', type=str, help='saved checkpoint directory', default=None)
    args = parser.parse_args()
    return args

def mainpulate(img, to_dir, origin_image, light_encoder, diffae, output_dir):
    to_dir = light_encoder.encode(to_dir)
    result = diffae.infer_manipulate(img, to_dir)
    trans = diffae.transforms(origin_image)
    trans = diffae.unnormalize(trans).permute(1, 2, 0).cpu().detach().numpy()

    fig, ax = plt.subplots(1, 3, tight_layout=True)
    for i, key in enumerate(['input', 'output']):
        ax[i].imshow(result[key])
        ax[i].set_title(key)
        ax[i].axis('off')
    ax[2].imshow(trans)
    ax[2].set_title('ground_truth')
    ax[2].axis('off')
    fig.savefig(output_dir + '/infer_result.png')

if __name__ == '__main__':
    args = vars(get_args())
    device = f'cuda:{args["device"]}'
    light_encoder = LightEncoder(in_dim=3, embed_level=1, include_input=True)
    mode = 'infer'
    diffae = DiffusionAutoEncodersInterface(args, mode)

    image_1 = Image.open(args['dataset_path'] + 'test/everett_dining1/dir_0_mip2.jpg')
    image_2 = Image.open(args['dataset_path'] + 'test//everett_dining1/dir_1_mip2.jpg')
    to_dir = torch.tensor([0.16421060785651204, -1.017222, 1.570796]).to(device) #24 image
    mainpulate(image_1, to_dir, image_2, light_encoder, diffae, args['output'])