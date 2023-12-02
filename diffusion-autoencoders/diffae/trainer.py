"""
The codes are modified.

Link:
    - [Trainer] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/core/trainer.py#L36-L382
"""
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
#from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from .models.loss import SimpleLoss, TripletLoss
from .utils import Meter, TimestepSampler, get_betas, seed_everything, training_reproducibility_cudnn

class Trainer:
    def __init__(self, model, cfg, output_dir, train_dataset):
        self.model = model
        self.cfg = cfg
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, **self.cfg['train']['dataloader'])

        self.device = self.cfg['general']['device']
        self.model.to(self.device)

        self.ckpt_dir = Path(self.output_dir / 'ckpt')
        self.ckpt_dir.mkdir(exist_ok=True)
        print(f'Checkpoints are saved in {self.ckpt_dir}')

        #self.tblogger = SummaryWriter(self.output_dir / 'tensorboard')
        print(f'Create a new event in {self.output_dir / "tensorboard"}')

        seed_everything(cfg['general']['seed'])
        training_reproducibility_cudnn()

        self.log_interval = cfg['train']['log_interval']
        self.save_interval = cfg['train']['save_interval']
        print(f'Output a log for every {self.log_interval} iteration')
        print(f'Save checkpoint every {self.save_interval} epoch')

        self.optimizer = self.get_optimizer()
        #self.scheduler = self.get_scheduler()
        self.criterion = SimpleLoss()
        self.triplet_loss = TripletLoss(self.device)

        self.fp16 = cfg['train']['fp16']
        self.grad_accum_steps = cfg['train']['grad_accum_steps']

        self.clip_grad_norm = cfg['train']['clip_grad_norm']

        self.num_timesteps = cfg['model']['timesteps']['num']
        self.timestep_sampler = TimestepSampler(cfg)

        self.betas = get_betas(cfg)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.mse_loss = torch.nn.MSELoss()
        self.light_position_coef = 100000
        self.triplet_loss_coef = 100000
        self.min_loss = 5000000
        self.loss_sum = 0
        self.loss_count = 0
        self.avg_loss_list = []

    def get_optimizer(self):
        optimizer_cfg = self.cfg['train']['optimizer']
        optimizer_cls = getattr(torch.optim, optimizer_cfg['name'])
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_cfg['params'])
        print(f'Use {optimizer_cfg["name"]} optimizer')
        return optimizer
    
    def get_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda t: min((t + 1) / 200, 1.0)
        )

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.cfg['train']['epoch']):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter, batch in enumerate(self.train_loader):
            self.before_iter()
            self.train_one_iter(batch)
            self.after_iter()

    def train_one_iter(self, batch):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            x0, light_source, label = batch
            x0 = x0.to(self.device)
            light_source = light_source.to(self.device)
            label = label.to(self.device)

            batch_size = x0.shape[0]
            t = self.timestep_sampler.sample(batch_size)

            noise = torch.randn_like(x0, device=self.device)
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise

            outputs, style_emb = self.model(x0, xt, t.float())
            l_position_err = self.mse_loss(light_source, style_emb[:, :9]) * self.light_position_coef
            triplet_loss = self.triplet_loss(style_emb[:,9:], label) * self.triplet_loss_coef
            noise_loss = self.criterion(outputs, noise)
            loss = noise_loss + l_position_err + triplet_loss
            loss /= self.grad_accum_steps

        self.scaler.scale(loss).backward()
        self.train_loss_meter.update([loss.item(), l_position_err.item(), triplet_loss.item(), noise_loss.item()])

        if (self.iter + 1) % self.grad_accum_steps == 0:
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         has_nan = torch.isnan(param.grad).any().item()
            #         has_inf = torch.isinf(param.grad).any().item()
            #         if has_nan or has_inf:
            #             print(f", Parameter: {name}, Grad Contains NaN: {has_nan}, Grad Contains Inf: {has_inf}")
            self.scaler.step(self.optimizer)
            #self.optimizer.step()
            #self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()


    def before_train(self):
        self.train_loss_meter = Meter()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        print('Training start ...')

    def after_train(self):
        print('Training done')

    def before_epoch(self):
        self.model.train()
        self.epoch_start_time = time()
        print(f'---> Start train epoch {self.epoch + 1}')

    def after_epoch(self):
        avg_loss = self.loss_sum / self.loss_count
        self.avg_loss_list.append(avg_loss)
        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self.save_ckpt(name=f'best_ckpt.pth')
        self.loss_sum = 0
        self.loss_count = 0
        epoch_elapsed_time = time() - self.epoch_start_time
        lr = self.optimizer.param_groups[0]['lr']
        print(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec). Avg loss:{avg_loss:.3f} LR: {lr:.4f}')

        self.plot_loss(f'loss_plot_no_grad_explode_triplet_loss_light_no_normalize_{self.light_position_coef}_{self.triplet_loss_coef}_no_batch_shuffle_light_encode_9.png')


    def before_iter(self):
        pass

    def after_iter(self):
        loss_pack = self.train_loss_meter.latest
        self.loss_sum += loss_pack[0]
        self.loss_count += 1
        if (self.iter + 1) % self.log_interval == 0:
            print(
                'epoch: {}/{}, iter: {}/{}, loss: {:.3f}, l_position: {:.3f}, triplet_loss: {:.3f}, l_noise: {:.3f}'.format(
                    self.epoch + 1, self.cfg['train']['epoch'],
                    self.iter + 1, len(self.train_loader),
                    loss_pack[0],
                    loss_pack[1],
                    loss_pack[2],
                    loss_pack[3],
                )
            )
            #self.tblogger.add_scalar('train_loss', self.train_loss_meter.latest, self.iter + 1)
            self.train_loss_meter.reset()

    def save_ckpt(self, name):
        print(f'Saving checkpoint to {self.ckpt_dir / name}')
        state = {
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.ckpt_dir / name)

    def plot_loss(self, name):
        print('Saving loss plot')
        plt.figure(figsize=(10,5))
        plt.title("Loss During Training")
        plt.plot(self.avg_loss_list,label="Loss")
        
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig(self.ckpt_dir / name)
