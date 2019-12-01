# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 20:50:56
@LastEditors: houwx
@LastEditTime: 2019-12-01 18:17:45
@Description: 
'''

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model.vqvae.vqvae import VQVAE
from model.modules import Audio2Mel

from model.melgan.melgan import Generator, Discriminator

class Trainer(object):
    def __init__(self, hps, train_data_loader, logger_path="ckpts/logs/", mode="vqvae"):
        self.hps = hps # Hyper-Parameters
        self.train_data_loader = train_data_loader # Mel Input Shape: (B, T, 80)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using cuda: ", torch.cuda.is_available())
        self.build_model()
        self.saved_model_list = []
        self.max_saved_model = hps.max_saved_model
        self.writer = SummaryWriter(logger_path)

    def build_model(self):
        self.audio2mel = Audio2Mel().to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=256, embed_dim=self.hps.vqvae_embed_dim, n_embed=self.hps.vqvae_n_embed).to(self.device)
        if self.mode == "vqvae":
            self.vqvae_optimizer = optim.Adam(self.vqvae.parameters(), lr=self.hps.lr)
        elif self.mode == "melgan": # Embeddings from VQVAE: (B, embed_dim, T_mel / 4)
            self.netG = Generator(self.hps.vqvae_n_embed, self.hps.ngf, self.hps.n_residual_layers).to(self.device)
            self.netD = Discriminator(self.hps.num_D, self.hps.ndf, self.hps.n_layers_D, self.hps.downsamp_factor)
            self.optG = torch.optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
            self.optD = torch.optim.Adam(self.netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        else:
            raise NotImplementedError("Invalid Mode!")
    

    def save_model(self, model_path, name, epoch, iterno, total_iterno, is_best_loss):
        if name == "vqvae":
            model = {'vqvae': self.vqvae.state_dict(),
                    'vqvae_optim': self.vqvae_optimizer.state_dict(),
                 }
        elif name == "melgan":
            model = {
				'generator': self.netG.state_dict(),
                'distriminator': self.netD.state_dict(),
                'generator_optim': self.optG.state_dict(),
                'discriminator_optim': self.optD.state_dict()
			}
        else:
            raise NotImplementedError("Invalid Model Name!")
        
        new_model_path = '{}-{}-{}-{}-{}-{}.pt'.format(model_path, name, epoch, iterno, total_iterno, is_best_loss)
        torch.save(model, new_model_path)
        self.saved_model_list.append(new_model_path)

        if len(self.saved_model_list) >= self.max_saved_model:
            os.remove(self.saved_model_list[0])
            self.saved_model_list.pop(0)
    

    def load_model(self, model_path, name):
        print('[Trainer] - load model from {}'.format(model_path))
        model = torch.load(model_path)
        if name == "vqvae":
            self.vqvae.load_state_dict(model['vqvae'])
            self.vqvae_optimizer.load_state_dict(model['vqvae_optim'])
        elif name == "mdelgan":
            self.netG.load_state_dict(model['generator'])
            self.netD.load_state_dict(model['discriminator'])
            self.optG.load_state_dict(model['generator_optim'])
            self.optD.load_state_dict(model['discriminator_optim'])
        else:
            raise NotImplementedError("Invalid Model Name!")
        print(f"{name} loaded.")


    def _clip_grad(self, net_list, max_grad_norm):
        for net in net_list:
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


    def train(self, save_model_path):
        total_iterno = -1
        if self.mode == "vqvae":
            loss_rec_best = 10000
            costs = []
            for epoch in range(1, self.hps.vqvae_epochs + 1):
                for iterno, (x, speaker_id) in enumerate(self.train_data_loader):
                    self.vqvae.train() # Set to train mode.
                    total_iterno += 1
                    x = x.to(self.device)
                    x_mel = self.audio2mel(x).detach().to(self.device)
                    x_rec, _ = self.vqvae(x_mel)
                    loss_rec = F.l1_loss(x_rec, x_mel)
                    loss_rec.backward()
                    self._clip_grad([self.vqvae], self.hps.max_grad_norm)
                    self.vqvae_optimizer.step()

                    # Update Tensorboard.
                    costs.append([loss_rec.item()])
                    self.writer.add_scalar(f'{self.mode}/loss_rec', costs[-1][0], total_iterno)

                    # Print info.
                    if iterno % self.hps.print_info_every == 0:
                        info = {f'{self.mode}/loss_rec': loss_rec.item(),
                                 '{self.mode}/loss_rec_mean': np.asarray(costs).mean(0),
                                }
                        slot_value = (epoch, self.hps.vqvae_epochs, iterno, len(self.train_data_loader), total_iterno) + tuple([value for value in info.values()])
                        log = 'VQVAE | Epochs: [%04d/%04d] | Iters:[%06d/%06d] | Total Iters: %d | Loss: %.3f | Mean Loss: %.3f'
                        print(log % slot_value)
                        # Clear costs.
                        costs = []                              
                    
                    # Save the checkpoint.
                    if (total_iterno > 1 and total_iterno % self.hps.save_model_every == 0) or (total_iterno > self.hps.start_save_best_model and loss_rec < loss_rec_best):
                        #loss_rec_best = min(loss_rec, loss_rec_best)
                        if loss_rec < loss_rec_best:
                            is_best_loss = True
                            loss_rec_best = loss_rec
                        else:
                            is_best_loss = False
                        self.save_model(model_path, self.mode, epoch, iterno, total_iterno, is_best_loss)
                        print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{total_iterno}, loss:{loss_rec}, is_best_loss:{is_best_loss}")
        
        elif self.mode == "melgan":
            pass # TODO    


if __name__ == "__main__":
    from dataset import AudioDataset
    from pathlib import Path
    import os
    from torch.utils.data import DataLoader
    from hps.hps import HyperParams
    Hps = HyperParams()
    hps = Hps.get_tuple()
    data_path = "../../databases/english/"
    rec_train_dataset = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=hps.seg_len, sampling_rate=22050)
    #test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
    train_data_loader = DataLoader(rec_train_dataset, batch_size=hps.batch_size, num_workers=4)
    #test_data_loader = DataLoader(test_set, batch_size=1)
    trainer = Trainer(hps=hps, train_data_loader=train_data_loader, mode="vqvae")
    model_path = os.path.join("ckpts", "model")
    trainer.train(save_model_path=model_path)
