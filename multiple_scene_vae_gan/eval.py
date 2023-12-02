import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
from dataloader2 import get_data_loader
from model_enlarged import VAE_GAN,Discriminator
from utils2 import show_and_save
import math


checkpoint = torch.load('./result/checkpoint_best.pth')

data_loader=get_data_loader(25)
gen=VAE_GAN().to(device)
discrim=Discriminator().to(device)

gen.load_state_dict(checkpoint['gen_state_dict'])
discrim.load_state_dict(checkpoint['discrim_state_dict'])

optim_E=torch.optim.Adam(gen.encoder.parameters(), lr=lr)
optim_D=torch.optim.Adam(gen.decoder.parameters(), lr=lr)
optim_Dis=torch.optim.Adam(discrim.parameters(), lr=lr*alpha)

optim_E.load_state_dict(checkpoint['optimizer_E_state_dict'])
optim_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
optim_Dis.load_state_dict(checkpoint['optimizer_Dis_state_dict'])

gen.eval()
discrim.eval()

b_list = []

for i, (data, light_source) in enumerate(data_loader, 0):
  light_source[:, :1] = (light_source[:, :1])-mean_value/std_dev
  show_and_save("target_%d" % i,make_grid((data*0.5+0.5).cpu(),8))
  datav = Variable(data).to(device)
  _, _, reconstructed_imgs, latent_v = gen(datav)
  for j in range (25):
    latent_v[:,:3] = light_source[j,:3]
    l_pos_shifted = gen.decoder(latent_v)
    show_and_save("Reconstructed Images_%d_position%d" % (i, j), make_grid((l_pos_shifted*0.5+0.5).cpu(), 8))



    
