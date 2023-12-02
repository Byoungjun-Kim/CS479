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
from dataloader import get_data_loader
from model_enlarged import VAE_GAN,Discriminator
from utils import show_and_save,plot_loss
import math

data_loader=get_data_loader(25, False)
gen=VAE_GAN().to(device)
discrim=Discriminator().to(device)
real_batch = next(iter(data_loader))
show_and_save("fixed_ground_truth" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))

epochs=3000
lr=1e-4
alpha=0.3
gamma=25

criterion=nn.BCELoss().to(device)
mse_loss = nn.MSELoss().to(device)
optim_E=torch.optim.Adam(gen.encoder.parameters(), lr=lr)
optim_D=torch.optim.Adam(gen.decoder.parameters(), lr=lr)
optim_Dis=torch.optim.Adam(discrim.parameters(), lr=lr*alpha)
z_fixed=Variable(torch.randn((64,128))).to(device)
x_fixed=Variable(real_batch[0]).to(device)
e_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_E, T_max=100, eta_min=0)
d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D, T_max=100, eta_min=0)
dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_Dis, T_max=100, eta_min=0)
gan_loss_list_t,recon_loss_list_t=[],[]
l_pos_list_t, l_var_list_t, dec_rec_list_t=[],[],[]

min_rec_err = float('inf')

for epoch in range(epochs):
  gan_loss_list,recon_loss_list=[],[]
  l_pos_list, l_var_list, dec_rec_list=[],[],[]

  for i, (data, light_source) in enumerate(data_loader, 0):
    bs=data.size()[0]

    ones_label=Variable(torch.ones(bs,1)).to(device)
    zeros_label=Variable(torch.zeros(bs,1)).to(device)
    zeros_label1=Variable(torch.zeros(64,1)).to(device)
    datav = Variable(data).to(device)
    mean, logvar, rec_enc, latent_v = gen(datav)

    z_p = Variable(torch.randn(64,128)).to(device)
    x_p_tilda = gen.decoder(z_p)
 
    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise 
    gan_loss_list.append(gan_loss.item())
    optim_Dis.zero_grad()
    gan_loss.backward(retain_graph=True)
    optim_Dis.step()

    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    # latent_rec_loss = 0
    # for j in range(bs):
    #   cur_light_invariance = latent_v[j,3:]
    #   new_z_tensor = torch.stack([torch.cat([light_source[k].to(device),cur_light_invariance]) for k in range(bs)])
    #   latent_rec_loss += mse_loss(gen.decoder(new_z_tensor), datav)
    # latent_rec_loss /= bs
    err_dec = gamma * rec_loss - gan_loss #+ latent_rec_loss
    recon_loss_list.append(rec_loss.item())
    optim_D.zero_grad()
    err_dec.backward(retain_graph=True)
    optim_D.step()
    
    mean, logvar, rec_enc, latent_v = gen(datav)
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
    l_position_err = mse_loss(light_source.to(device), latent_v[:, :3])
    l_invariant_err = torch.var(latent_v[:,3:], dim=0).sum()
    err_enc = prior_loss + 10*rec_loss + 100*l_position_err + 25*l_invariant_err
    optim_E.zero_grad()
    err_enc.backward(retain_graph=True)
    optim_E.step()
    e_scheduler.step()
    d_scheduler.step()
    dis_scheduler.step()

  print('[%d/%d]\tLoss_gan: %.4f\tRec_loss: %.4f\tl_pos_loss: %0.4f'
        % (epoch,epochs,
            gan_loss.item(),rec_loss.item(),l_position_err.item()))

  b=gen(x_fixed)[2]
  b=b.detach()
  show_and_save('fixed_output_epoch_%d' % epoch ,make_grid((b*0.5+0.5).cpu(),8))

  rec_err_avg = sum(recon_loss_list)/len(recon_loss_list)
  gan_loss_list_t.append((sum(gan_loss_list)/len(gan_loss_list)))
  recon_loss_list_t.append(rec_err_avg)

  if (rec_err_avg<=min_rec_err):
    min_rec_err = rec_err_avg
    checkpoint = {
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'discrim_state_dict': discrim.state_dict(),
        'optimizer_E_state_dict': optim_E.state_dict(),
        'optimizer_D_state_dict': optim_D.state_dict(),
        'optimizer_Dis_state_dict': optim_Dis.state_dict(),
        'scheduler_E_state_dict': e_scheduler.state_dict(),
        'scheduler_D_state_dict': d_scheduler.state_dict(),
        'scheduler_Dis_state_dict': dis_scheduler.state_dict(),
        'recon_loss_list_t': recon_loss_list_t,
        'gan_loss_list_t': gan_loss_list_t    
    }
    torch.save(checkpoint, f'./result/checkpoint_best.pth')
    print('best epoch %d', epoch)

  plot_loss(recon_loss_list_t, 'recon')
  plot_loss(gan_loss_list_t, 'gan')
