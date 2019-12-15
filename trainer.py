# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 20:50:56
@LastEditors: houwx
@LastEditTime: 2019-12-08 16:54:48
@Description: 
'''
import time

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model.vqvae.vqvae import VQVAE
from model.modules import Audio2Mel

from model.melgan.melgan import Generator, Discriminator

torch.manual_seed(1)

class Trainer(object):
    def __init__(self, hps, train_data_loader, logger_path="ckpts/logs/", mode="vqvae", num_speaker=-1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using cuda: ", torch.cuda.is_available())

        self.hps = hps # Hyper-Parameters
        #self.train_data_loader = train_data_loader # Mel Input Shape: (B, T, 80)
        self.train_data_loader = train_data_loader # Use tqdm progress bar
        self.mode = mode
        
        self.num_speaker = num_speaker
        if mode == "vqvae":
            self.lr = self.hps.lr_vqvae # Initial learning rate
        elif mode == "melgan":
            self.lr = self.hps.lr_melgan

        self.saved_model_list = []
        self.best_model_list = []
        self.max_saved_model = hps.max_saved_model # Max number of saved model.
        self.max_best_model = hps.max_best_model # Max number of model with best losses.
        
        #self.writer = SummaryWriter(logger_path) # Tensoboard Writer
        
        self.criterion = nn.MSELoss() # Use MSE Loss
        self.build_model()

        self.accum_epochs = 0 # For reload training.
        self.accum_iterno = 0 # For reload training.

    def build_model(self):
        self.audio2mel = Audio2Mel().to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=512, 
                            embed_dim=self.hps.vqvae_embed_dim, 
                            n_embed=self.hps.vqvae_n_embed,
                            num_speaker=self.num_speaker,
                            ).to(self.device)
        if self.mode == "vqvae":
            self.vqvae_optimizer = optim.Adam(self.vqvae.parameters(), lr=self.lr)
        elif self.mode == "melgan": # Embeddings from VQVAE: (B, embed_dim, T_mel / 4)
            self.netG = Generator(2 * self.hps.vqvae_embed_dim, self.hps.ngf, self.hps.n_residual_layers).to(self.device)
            self.netD = Discriminator(self.hps.num_D, self.hps.ndf, self.hps.n_layers_D, self.hps.downsamp_factor)
            self.optG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.9))
            self.optD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.9))
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

        if not is_best_loss:
            self.saved_model_list.append(new_model_path)
            if len(self.saved_model_list) > self.max_saved_model:
                os.remove(self.saved_model_list[0])
                self.saved_model_list.pop(0)
        else: # Is best loss
            self.best_model_list.append(new_model_path)
            if len(self.best_model_list) > self.max_best_model:
                os.remove(self.best_model_list[0])
                self.best_model_list.pop(0)
        

    def load_model(self, model_file, name):
        print('[Trainer] - load model from {}'.format(model_file))
        model = torch.load(model_file)
        if name == "vqvae":
            self.vqvae.load_state_dict(model['vqvae'])
            self.vqvae_optimizer.load_state_dict(model['vqvae_optim'])
        elif name == "vqvae/encoder_only":
            model_dict=self.vqvae.state_dict()
            # Filter out decoder's keys in state dict
            load_dict = {k: v for k, v in model['vqvae'].items() if k in model_dict and "dec" not in k}
            model_dict.update(load_dict)
            self.vqvae.load_state_dict(model_dict)
        elif name == "mdelgan":
            self.netG.load_state_dict(model['generator'])
            self.netD.load_state_dict(model['discriminator'])
            self.optG.load_state_dict(model['generator_optim'])
            self.optD.load_state_dict(model['discriminator_optim'])
        else:
            raise NotImplementedError("Invalid Model Name!")
        
        import os
        self.accum_epochs = int(os.path.basename(model_file).split('-')[2]) - 1
        self.accum_iterno = int(os.path.basename(model_file).split('-')[4])
        print(f"{name} loaded.")


    def _clip_grad(self, net_list, max_grad_norm):
        for net in net_list:
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    def _halve_lr(self, optimizer):
        self.lr /= 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, save_model_path):
        total_iterno = -1
        if self.mode == "vqvae":
            loss_rec_best = 10000
            costs = []
            start = time.time()
            for epoch in range(1, self.hps.vqvae_epochs + 1):
                train_data_loader = tqdm(self.train_data_loader, total=len(self.train_data_loader))
                for iterno, (x, speaker_id) in enumerate(train_data_loader):
                    self.vqvae.train() # Set to train mode.
                    total_iterno += 1
                    x = x.to(self.device)
                    x_mel = self.audio2mel(x).detach()
                    if self.num_speaker != -1:
                        speaker_id = speaker_id.to(self.device)
                        x_rec, loss_latent = self.vqvae(x_mel.to(self.device), speaker_id=speaker_id)
                    else:
                        x_rec, loss_latent = self.vqvae(x_mel.to(self.device))
                    # loss_latent: commitment loss, to make encodings get close to codebooks (quantize.detach() - input).pow(2).mean()
                    loss_rec = self.criterion(x_rec, x_mel) # Loss of reconstruction.
                    # Impotant: loss should be the combination of reconstruction loss and commitment loss in VQVAE.
                    loss_vqvae = loss_rec + self.hps.loss_latent_weight * loss_latent

                    # Reset gradients.
                    self.vqvae.zero_grad()
                    loss_vqvae.backward()
                    self._clip_grad([self.vqvae], self.hps.max_grad_norm)
                    self.vqvae_optimizer.step()

                    # For reloaded model training.
                    accum_iterno = self.accum_iterno + total_iterno
                    accum_epochs = self.accum_epochs + epoch
                    # Schedule learning rate.
                    if accum_iterno in [4e5, 6e5, 8e5]:
                        self._halve_lr(self.vqvae_optimizer)

                    # Save losses
                    costs.append([loss_rec.item(), loss_latent.item()])
                    mean_loss_rec = np.array(costs).mean(0)[0]
                    mean_loss_latent = np.array(costs).mean(0)[1]
                    # Update Tensorboard.
                    info = {
                        f'{self.mode}/loss_rec': costs[-1][0],
                        f'{self.mode}/loss_latent': costs[-1][1],
                        f'{self.mode}/mean_loss_rec': mean_loss_rec,
                        f'{self.mode}/mean_loss_latent': mean_loss_latent,
                    }
                    '''
                    for tag, value in info.items():
                        self.writer.add_scalar(tag, value, total_iterno)
                    '''
                    train_data_loader.set_description(
                        (
                            f'epochs: {accum_iterno}; loss_rec: {costs[-1][0]}; loss_latent: {costs[-1][1]}; mean_loss_rec: {mean_loss_rec}'
                        )
                    )
                    # Print info.
                    if total_iterno > 0 and total_iterno % self.hps.print_info_every == 0:
                        
                        slot_value = (accum_epochs, self.hps.vqvae_epochs, iterno, len(self.train_data_loader), accum_iterno) + tuple([value for value in info.values()][-2:]) + \
                                                        tuple([1000 * (time.time() - start) / self.hps.print_info_every])
                        log = 'VQVAE Stats | Epochs: [%04d/%04d] | Iters:[%06d/%06d] | Total Iters: %d | Mean Rec Loss: %.3f | Mean Latent Loss: %.3f | Ms/Batch: %5.2f'
                        #print(log % slot_value)
                        train_data_loader.write(log % slot_value)
                        # Clear costs.
                        costs = []
                        start = time.time()
                    
                    # Save the checkpoint.
                    if total_iterno > 0 and total_iterno % self.hps.save_model_every == 0:
                        #loss_rec_best = min(loss_rec, loss_rec_best)
                        # Save best models
                        if total_iterno > self.hps.start_save_best_model and mean_loss_rec < loss_rec_best:
                            loss_rec_best = mean_loss_rec
                            is_best_loss = True
                            self.save_model(model_path, self.mode, accum_epochs, iterno, accum_iterno, is_best_loss)
                            #print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{self.total_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                            train_data_loader.write(f"Model saved: {self.mode}, epoch: {accum_epochs}, iterno: {iterno}, total_iterno:{accum_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                        else:
                            is_best_loss = False
                            self.save_model(model_path, self.mode, accum_epochs, iterno, accum_iterno, is_best_loss)
                            #print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{self.total_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                            train_data_loader.write(f"Model saved: {self.mode}, epoch: {accum_epochs}, iterno: {iterno}, total_iterno:{accum_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                    #train_data_loader.update()
                    train_data_loader.write("-" * 100)

        elif self.mode == "melgan":
            loss_rec_best = 10000
            costs = []
            start = time.time()

            self.vqvae.eval()
            for epoch in range(1, self.hps.melgan_epochs + 1):
                train_data_loader = tqdm(self.train_data_loader, total=len(self.train_data_loader))
                for iterno, (x, speaker_id) in enumerate(train_data_loader):
                    total_iterno += 1
                    x = x.to(self.device)
                    x_mel = self.audio2mel(x).detach()
                    quant_t, quant_b, _, _, _ = self.vqvae.encode(x_mel)
                    x_enc = self.vqvae.get_encoding(quant_t, quant_b).detach() # x_enc: (B, 2 * embed_dim, T/4)
                    x_pred = self.netG(x_enc.to(self.device))

                    with torch.no_grad():
                        x_pred_mel = self.audio2mel(x_pred.detach())
                        enc_error = F.l1_loss(x_mel, x_pred_mel).item()
                    
                    #######################
                    # Train Discriminator #
                    #######################
                    D_fake_det = self.netD(x_pred.to(self.device).detach())
                    D_real = self.netD(x.to(self.device))

                    loss_D = 0
                    for scale in D_fake_det:
                        loss_D += F.relu(1 + scale[-1]).mean()

                    for scale in D_real:
                        loss_D += F.relu(1 - scale[-1]).mean()

                    self.netD.zero_grad()
                    loss_D.backward()
                    self.optD.step()

                    ###################
                    # Train Generator #
                    ###################
                    D_fake = self.netD(x_pred.to(self.device))

                    loss_G = 0
                    for scale in D_fake:
                        loss_G += -scale[-1].mean()

                    loss_feat = 0
                    feat_weights = 4.0 / (self.hps.n_layers_D + 1)
                    D_weights = 1.0 / self.hps.num_D
                    wt = D_weights * feat_weights
                    for i in range(self.hps.num_D):
                        for j in range(len(D_fake[i]) - 1):
                            loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

                    self.netG.zero_grad()
                    (loss_G + self.hps.lambda_feat * loss_feat).backward()
                    self.optG.step()

                    ######################
                    # Update tensorboard #
                    ######################
                    # For reloaded model training.
                    accum_iterno = self.accum_iterno + total_iterno
                    accum_epochs = self.accum_epochs + epoch

                    costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), enc_error])
                    '''
                    self.writer.add_scalar("melgan_loss/discriminator", costs[-1][0], total_iterno)
                    self.writer.add_scalar("melgan_loss/generator", costs[-1][1], total_iterno)
                    self.writer.add_scalar("melgan_loss/feature_matching", costs[-1][2], total_iterno)
                    self.writer.add_scalar("melgan_loss/enc_reconstruction", costs[-1][3], total_iterno)
                    '''

                    mean_loss_rec = np.array(costs).mean(0)[-1]
                    mean_D_loss = np.array(costs).mean(0)[0]
                    mean_G_loss = np.array(costs).mean(0)[1]
                    info = {
                        f'{self.mode}/D_loss': costs[-1][0],
                        f'{self.mode}/G_loss': costs[-1][1],
                        f'{self.mode}/feature_matching_loss': costs[-1][2],
                        f'{self.mode}/loss_rec': costs[-1][3],
                        f'{self.mode}/mean_D_loss': mean_D_loss,
                        f'{self.mode}/mean_G_loss': mean_G_loss,
                        f'{self.mode}/mean_loss_rec': mean_loss_rec,
                    }
                    '''
                    for tag, value in info.items():
                        self.writer.add_scalar(tag, value, total_iterno)
                    '''
                    train_data_loader.set_description(
                        (
                            f'epochs: {accum_iterno}; loss_rec: {costs[-1][0]}; loss_latent: {costs[-1][1]}; mean_loss_rec: {mean_loss_rec}'
                        )
                    )
                    # Print info.
                    if total_iterno > 0 and total_iterno % self.hps.print_info_every == 0:
                        
                        slot_value = (accum_epochs, self.hps.vqvae_epochs, iterno, len(self.train_data_loader), accum_iterno) + tuple([value for value in info.values()][-3:]) + \
                                                        tuple([1000 * (time.time() - start) / self.hps.print_info_every])
                        log = 'VQVAE Stats | Epochs: [%04d/%04d] | Iters:[%06d/%06d] | Total Iters: %d | Mean D Loss: %.3f | Mean G Loss: %.3f | Mean Rec Loss: %.3f | Ms/Batch: %5.2f'
                        #print(log % slot_value)
                        train_data_loader.write(log % slot_value)
                        # Clear costs.
                        costs = []
                        start = time.time()

                    # Save the checkpoint.
                    if total_iterno > 0 and total_iterno % self.hps.save_model_every == 0:
                        # Save best models
                        if total_iterno > self.hps.start_save_best_model and mean_loss_rec < loss_rec_best:
                            loss_rec_best = mean_loss_rec
                            is_best_loss = True
                            self.save_model(model_path, self.mode, accum_epochs, iterno, accum_iterno, is_best_loss)
                            #print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{self.total_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                            train_data_loader.write(f"Model saved: {self.mode}, epoch: {accum_epochs}, iterno: {iterno}, total_iterno:{accum_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                        else:
                            is_best_loss = False
                            self.save_model(model_path, self.mode, accum_epochs, iterno, accum_iterno, is_best_loss)
                            #print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{self.total_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                            train_data_loader.write(f"Model saved: {self.mode}, epoch: {accum_epochs}, iterno: {iterno}, total_iterno:{accum_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                    #train_data_loader.update()
                    train_data_loader.write("-" * 100)


if __name__ == "__main__":
    from dataset import AudioDataset
    from pathlib import Path
    import os
    from torch.utils.data import DataLoader
    from hps.hps import HyperParams
    Hps = HyperParams()
    hps = Hps.get_tuple()
    #data_path = "../../databases/english/" # On lab server.
    data_path = "./databases/english_small/" # On my own PC.
    mode = "melgan"
    
    load_vqvae = False
    if mode == "vqvae":
        rec_train_dataset = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=hps.seg_len, sampling_rate=16000)
        num_speaker = rec_train_dataset.get_speaker_num()
        #test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
        train_data_loader = DataLoader(rec_train_dataset, batch_size=hps.batch_size, shuffle=True, num_workers=4)#hps.batch_size, num_workers=4)
        #test_data_loader = DataLoader(test_set, batch_size=1)
        trainer = Trainer(hps=hps, train_data_loader=train_data_loader, mode=mode, num_speaker=num_speaker)
    
    elif mode == "melgan":
        load_vqvae = True
        gan_train_dataset = AudioDataset(audio_files=Path(data_path) / "gan_train_files.txt", segment_length=hps.seg_len, sampling_rate=16000)
        num_speaker = gan_train_dataset.get_speaker_num()
        #test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
        train_data_loader = DataLoader(gan_train_dataset, batch_size=hps.batch_size, shuffle=True, num_workers=4)#hps.batch_size, num_workers=4)
        #test_data_loader = DataLoader(test_set, batch_size=1)
        trainer = Trainer(hps=hps, train_data_loader=train_data_loader, mode=mode, num_speaker=-1)
    
    model_path = os.path.join("ckpts", "model")
    
    if load_vqvae:
        name = 'vqvae' if mode == "vqvae" else 'vqvae/encoder_only'
        trainer.load_model('ckpts/model-vqvae-3-0-4-True.pt', name)
    trainer.train(save_model_path=model_path)
