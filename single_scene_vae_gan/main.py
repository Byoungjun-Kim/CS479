import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
from dataloader import get_data_loader
from models.vae_gan import VAE_GAN,Discriminator
from utils import show_and_save,plot_loss

data_loader,test_loader=get_data_loader(20)
gen=VAE_GAN().to(device)
discrim=Discriminator().to(device)
real_batch = next(iter(data_loader))
show_and_save("training" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))

def loss_function(criterion, recon_x,x,mean,logstd):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    MSE = criterion(recon_x,x)
    var = torch.pow(torch.exp(logstd),2)
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
    return MSE+KLD
    
trial = 24
epochs=2001
lr=3e-5
alpha=0.2
gamma=15
light_source_coefficient = 1000
fp = open(f'./res/{trial}/output.txt', 'w')

criterion=nn.BCELoss().to(device)
mse_loss = nn.MSELoss().to(device)
optim_E=torch.optim.Adam(gen.encoder.parameters(), lr=lr)
optim_D=torch.optim.Adam(gen.decoder.parameters(), lr=lr)
optim_Dis=torch.optim.Adam(discrim.parameters(), lr=lr*alpha)

e_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_E, T_max=epochs, eta_min=0)
d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D, T_max=epochs, eta_min=0)
dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_Dis, T_max=epochs, eta_min=0)

z_fixed=Variable(torch.randn((64,128))).to(device)
x_fixed=Variable(real_batch[0]).to(device)
light_source_fixed = Variable(real_batch[1]).to(device)

for epoch in range(epochs):
    prior_loss_list,gan_loss_list,recon_loss_list=[],[],[]
    dis_real_list,dis_fake_list,dis_prior_list=[],[],[]
    for i, (data, light_source) in enumerate(data_loader, 0):
        bs=data.size()[0]

        ones_label=Variable(torch.ones(bs,1)).to(device)
        zeros_label=Variable(torch.zeros(bs,1)).to(device)
        datav = Variable(data).to(device)
        light_source = Variable(light_source).to(device)
        mean, logvar, z, rec_enc = gen(datav)
        z_light_pos, z_invariant_feature = z[:,:3], z[:,3:]
        light_err = mse_loss(light_source, z_light_pos)
        feature_variance = torch.sum((z_invariant_feature - torch.mean(z_invariant_feature,dim=0).unsqueeze(0))**2)/bs 
        z_p = Variable(torch.randn(bs,128)).to(device)
        x_p_tilda = gen.decoder(z_p)
        latent_rec_loss = 0
        for i in range(bs):
            cur_light_invariance = z[i,3:]
            new_z_tensor = torch.stack([torch.cat([light_source[j],cur_light_invariance]) for j in range(bs)])
            latent_rec_loss += mse_loss(gen.decoder(new_z_tensor), datav)
        latent_rec_loss /= bs
        output = discrim(datav)[0]
        errD_real = criterion(output, ones_label)
        dis_real_list.append(errD_real.item())
        output = discrim(rec_enc)[0]
        errD_rec_enc = criterion(output, zeros_label)
        dis_fake_list.append(errD_rec_enc.item())
        output = discrim(x_p_tilda)[0]
        errD_rec_noise = criterion(output, zeros_label)
        dis_prior_list.append(errD_rec_noise.item())
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
        errD_rec_noise = criterion(output, zeros_label)
        gan_loss = errD_real + errD_rec_enc + errD_rec_noise
        x_l_tilda = discrim(rec_enc)[1]
        x_l = discrim(datav)[1]
        rec_loss = ((x_l_tilda - x_l) ** 2).mean()
        err_dec = gamma * rec_loss + latent_rec_loss - gan_loss 
        recon_loss_list.append(rec_loss.item())
        optim_D.zero_grad()
        err_dec.backward(retain_graph=True)
        optim_D.step()
        mean, logvar, z, rec_enc = gen(datav)
        x_l_tilda = discrim(rec_enc)[1]
        x_l = discrim(datav)[1]
        rec_loss = ((x_l_tilda - x_l) ** 2).mean()
        prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
        prior_loss_list.append(prior_loss.item())
        err_enc = prior_loss + rec_loss + light_err + 4*feature_variance
        optim_E.zero_grad()
        err_enc.backward(retain_graph=True)
        optim_E.step()
        e_scheduler.step()
        d_scheduler.step()
        dis_scheduler.step()

    print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
            % (epoch,epochs, i, len(data_loader),
                gan_loss.item(), prior_loss.item(),rec_loss.item(),errD_real.item(),errD_rec_enc.item(),errD_rec_noise.item()))
    print('[%d/%d][%d/%d]\tlight_err: %.4f\tfeature_variance: %.4f\tlatent_loss: %.4f'
                % (epoch,epochs, i, len(data_loader),
                    light_err,feature_variance,latent_rec_loss))
    if epoch % 10 == 0:
        for (d, ls) in train_loader:
            validate_data = Variable(d).to(device)
            ls = ls.to(device)
            _,_,z,b= gen(datav)
            _,_,vz,vb= gen(validate_data)
            ground_loss = mse_loss(validate_data,vb)/5
            fp.write(f"{epoch} ground_loss : {ground_loss}\n")
            rec_loss = 0
            for i in range(5):
                z[:,:3] = ls[i,:]
                rec_loss += mse_loss(gen.decoder(z),datav)
            rec_loss /= 5
            fp.write(f"{epoch} rec_loss : {rec_loss}\n")

checkpoint = {
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'discrim_state_dict': discrim.state_dict(),
        'optimizer_E_state_dict': optim_E.state_dict(),
        'optimizer_D_state_dict': optim_D.state_dict(),
        'optimizer_Dis_state_dict': optim_Dis.state_dict(),
        'prior_loss_list': prior_loss_list,
        'recon_loss_list': recon_loss_list,
        'gan_loss_list': gan_loss_list
    }

torch.save(checkpoint, f'./res/{trial}/checkpointepoch{epoch}.pth')