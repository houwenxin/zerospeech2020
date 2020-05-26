# -*- coding: UTF-8 -*-

import time
from dataset import AudioDataset
from pathlib import Path
import os
from torch.utils.data import DataLoader
from hps import HyperParams
import argparse
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable, detect_anomaly
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model.vqvae.vqvae import VQVAE
from model.modules import Audio2Mel

from model.melgan.melgan import Generator, Discriminator

seed = 12321
torch.manual_seed(seed) # 1 VQVAE + MelGAN before model-melgan-937-441-414153-False.pt
# 0 also for MelGAN pretraining. 1 for second time

#torch.manual_seed(seed)  # 1234 MelGAN after model-melgan-937-441-414153-False.pt

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
        

        self.writer = SummaryWriter(logger_path) # Tensoboard Writer
        
        self.criterion = nn.L1Loss() # Try L1 Loss. # nn.MSELoss() # Use MSE Loss
        self.build_model()

        self.accum_epochs = 0 # For reload training.
        self.accum_iterno = 0 # For reload training.

    def build_model(self):
        self.audio2mel = Audio2Mel(hop_length=self.hps.hop_length).to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=512, 
                            embed_dim=self.hps.vqvae_embed_dim, # 64
                            n_embed=self.hps.vqvae_n_embed, # 128
                            num_speaker=self.num_speaker,
                            ).to(self.device)

        if self.mode == "vqvae":
            self.vqvae_optimizer = optim.Adam(self.vqvae.parameters(), lr=self.lr)

        elif self.mode == "melgan": # Embeddings from VQVAE: (B, embed_dim, T_mel / 4)
            input_dim = 2 * self.hps.vqvae_embed_dim
            if self.num_speaker != -1:
                self.spk_embed = nn.Embedding(self.num_speaker, self.hps.vqvae_embed_dim // 2).to(self.device)
                input_dim += self.hps.vqvae_embed_dim // 2
            self.netG = Generator(input_dim, self.hps.ngf, self.hps.n_residual_layers).to(self.device)
            self.netD = Discriminator(self.hps.num_D, self.hps.ndf, self.hps.n_layers_D, self.hps.downsamp_factor).to(self.device)

            if self.num_speaker != -1:
                self.optG = optim.Adam(list(self.netG.parameters()) + list(self.spk_embed.parameters()), lr=self.lr, betas=(0.5, 0.9))
            else:
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
                'discriminator': self.netD.state_dict(),
                'generator_optim': self.optG.state_dict(),
                'discriminator_optim': self.optD.state_dict()
			}
            if self.num_speaker != -1:
                model['spk_embed'] = self.spk_embed.state_dict()
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
            load_dict = {k: v for k, v in model['vqvae'].items() if k in model_dict and "dec" not in k and "spk" not in k}
            model_dict.update(load_dict)
            self.vqvae.load_state_dict(model_dict)
        elif name == "melgan":
            self.netG.load_state_dict(model['generator'])
            self.netD.load_state_dict(model['discriminator'])
            self.optG.load_state_dict(model['generator_optim'])
            self.optD.load_state_dict(model['discriminator_optim'])
            if self.num_speaker != -1:
                self.spk_embed.load_state_dict(model['spk_embed'])
        else:
            raise NotImplementedError("Invalid Model Name!")
        
        if self.mode in name or name in self.mode:
            import os
            self.accum_epochs = int(os.path.basename(model_file).split('-')[2])# - 1
            self.accum_iterno = int(os.path.basename(model_file).split('-')[4])
        print(f"{name} loaded.")


    def _clip_grad(self, net_list, max_grad_norm):
        for net in net_list:
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    def _halve_lr(self, optimizer):
        self.lr /= 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def _fix_model_params(self, net_list):
        # IMPORTANT: Do not require gradients for Audio2Mel, or it will cause NaN in back propagration (sqrt).
        for net in net_list:
            print(f"Fix parameters of {net.__class__.__name__}")
            for p in net.parameters():
                p.requires_grad = False

    def train(self, save_model_path, test_data_loader=None):
        total_iterno = -1
        # enable cudnn autotuner to speed up training
        torch.backends.cudnn.benchmark = True

        if self.mode == "vqvae":
            loss_rec_best = 10000
            costs = []
            start = time.time()
            
            if self.accum_iterno >= 4e6:
                self._halve_lr(self.vqvae_optimizer)
            if self.accum_iterno >= 6e6:
                self._halve_lr(self.vqvae_optimizer)
            if self.accum_iterno >= 8e6:
                self._halve_lr(self.vqvae_optimizer)

            for epoch in range(1, self.hps.vqvae_epochs + 1):
                if self.accum_epochs == self.hps.vqvae_epochs:
                    break
                train_data_loader = tqdm(self.train_data_loader, total=len(self.train_data_loader))
                for iterno, (x, speaker_info) in enumerate(train_data_loader):
                    self.vqvae.train() # Set to train mode.
                    total_iterno += 1
                    x = x.to(self.device)
                    x_mel = self.audio2mel(x).detach()
                    if self.num_speaker != -1:
                        speaker_id = speaker_info['source_speaker']['id'].to(self.device)
                        x_rec, loss_latent, x_enc = self.vqvae(x_mel.to(self.device), speaker_id=speaker_id)
                    else:
                        x_rec, loss_latent, x_enc = self.vqvae(x_mel.to(self.device))
                    # loss_latent: commitment loss, to make encodings get close to codebooks (quantize.detach() - input).pow(2).mean()
                    loss_rec = self.criterion(x_rec, x_mel) # Loss of reconstruction.
                    # Impotant: loss should be the combination of reconstruction loss and commitment loss in VQVAE.
                    loss_vqvae = loss_rec + self.hps.loss_latent_weight * loss_latent


                    # For reloaded model training.
                    accum_iterno = self.accum_iterno + total_iterno
                    accum_epochs = self.accum_epochs + epoch
                    
                    update_vqvae = True
                    if update_vqvae:
                        # Reset gradients.
                        self.vqvae.zero_grad()
                        loss_vqvae.backward()
                        self._clip_grad([self.vqvae], self.hps.max_grad_norm)
                        self.vqvae_optimizer.step()

                        # Schedule learning rate.
                        if accum_iterno in [4e6, 6e6, 8e6]: #[4e5, 6e5, 8e5]:
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
                    
                    for tag, value in info.items():
                        self.writer.add_scalar(tag, value, accum_iterno)
                    
                    train_data_loader.set_description(
                        (
                            f'total_epoch: {accum_epochs}; total_iterno: {accum_iterno}; mean_loss_rec: {mean_loss_rec};'
                        )
                    )
                        
                    # Print info.
                    if total_iterno > 0 and total_iterno % self.hps.print_info_every == 0:
                        
                        slot_value = (accum_epochs, self.hps.vqvae_epochs, iterno, len(self.train_data_loader), accum_iterno) + tuple([info[f"{self.mode}/mean_loss_rec"], info[f"{self.mode}/mean_loss_latent"]]) + \
                                                        tuple([1000 * (time.time() - start) / self.hps.print_info_every])
                        log = 'VQVAE Stats | Epochs: [%05d/%05d] | Iters: [%05d/%05d] | Total Iters: %d | Mean Rec Loss: %.3f | Mean Latent Loss: %.3f | Ms/Batch: %5.2f'
                        #print(log % slot_value)
                        train_data_loader.write(log % slot_value)
                        # Clear costs.
                        costs = []
                        start = time.time()
                    
                # Save the checkpoint.
                #if total_iterno > 0 and total_iterno % self.hps.save_model_every == 0:
                    #loss_rec_best = min(loss_rec, loss_rec_best)
                    # Save best models
                if epoch % self.hps.valid_every_n_epoch == 0:
                    if test_data_loader:
                        train_data_loader.write("Evaluating mean reconstruction loss on test data...")
                        mean_loss_rec = self.evaluate_vqvae(test_data_loader)

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
                
                train_data_loader.write("-" * 200)

        elif self.mode == "melgan":
            loss_mel_best = 10000
            costs = []
            start = time.time()

            self.vqvae.eval()
            self.netG.train()
            self.netD.train()
            self._fix_model_params([self.vqvae, self.audio2mel])
            #self.spk_embed.train()

            if self.accum_iterno >= 4e5:
                self._halve_lr(self.optD)
                self._halve_lr(self.optG)
            if self.accum_iterno >= 6e5:
                self._halve_lr(self.optD)
                self._halve_lr(self.optG)
            if self.accum_iterno >= 8e5:
                self._halve_lr(self.optD)
                self._halve_lr(self.optG)
            
            with detect_anomaly():
                for epoch in range(1, self.hps.melgan_epochs + 1):
                    if self.accum_epochs == self.hps.melgan_epochs:
                        break
                    train_data_loader = tqdm(self.train_data_loader, total=len(self.train_data_loader))
                    for iterno, (x, speaker_info) in enumerate(train_data_loader):
                        total_iterno += 1
                        x = x.to(self.device)
                        x_mel = self.audio2mel(x).detach()

                        quant_t, quant_b, _, _, _ = self.vqvae.encode(x_mel)
                        x_enc_org = self.vqvae.get_encoding(quant_t.detach(), quant_b.detach()).detach() # x_enc: (B, 2 * embed_dim, T/4)

                        assert not np.isnan(x_enc_org.detach().cpu()).any(), "Find NaN in VQVAE Encoding!" # Assert not to contain NaN in inputs. Meet NaN loss problem during training.
                        
                        if self.num_speaker != -1:
                            speaker_id = speaker_info['source_speaker']['id'].to(self.device)
                            assert (speaker_id < self.num_speaker).all(), "Speaker ID should be smaller than the total number of speakers"
                            assert -1 not in speaker_id, "-1 appears in speaker ID."

                            spk = self.spk_embed(speaker_id)

                            spk = spk.view(spk.size(0), spk.size(1), 1)  # Return: (B, embed_dim // 2, 1)
                            spk_expand = spk.expand(spk.size(0), spk.size(1), x_enc_org.size(2)) # Return: (B, embed_dim // 2, T / 4)
                            #print("\n", spk_expand.shape)
                            x_enc = torch.cat((x_enc_org, spk_expand), dim=1) 
                        
                            assert not np.isnan(spk_expand.detach().cpu()).any(), "Find NaN in Speaker Embedding!" # Assert not to contain NaN in inputs. Meet NaN loss problem during training.

                        x_pred = self.netG(x_enc.to(self.device))
                        x_pred_mel = self.audio2mel(x_pred)
                        mel_error = self.criterion(x_mel, x_pred_mel)

                        quant_t, quant_b, _, _, _ = self.vqvae.encode(x_pred_mel)
                        x_pred_enc_org = self.vqvae.get_encoding(quant_t, quant_b) # x_enc: (B, 2 * embed_dim, T/4)
                        enc_error = self.criterion(x_enc_org, x_pred_enc_org)

                        with torch.no_grad():
                            rec_error = self.criterion(x, x_pred)
                            
                        # print(torch.max(x_pred))
                        # print(torch.max(x))
                        #for param in self.netD.parameters():
                        #    print(torch.max(x))
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
                        # self._clip_grad([self.netD], 1.0) # TODO: handle grad explosion
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
                                loss_feat += wt * self.criterion(D_fake[i][j], D_real[i][j].detach())
                                #loss_feat += wt * self.criterion(D_fake[i][j], D_real[i][j].detach())

                        self.netG.zero_grad()
                        if self.num_speaker != -1:
                            self.spk_embed.zero_grad() # Update along with Embedding layer.
                        (self.hps.adv_loss_weight * loss_G + self.hps.lambda_feat * loss_feat + self.hps.enc_error_weight * enc_error + self.hps.mel_error_weight * mel_error).backward()
                        #self._clip_grad([self.netG, self.spk_embed], self.hps.max_grad_norm) # TODO: handle grad explosion
                        self.optG.step()
                        

                        ######################
                        # Update tensorboard #
                        ######################
                        # For reloaded model training.
                        accum_iterno = self.accum_iterno + total_iterno
                        accum_epochs = self.accum_epochs + epoch


                        # Schedule learning rate.
                        if accum_iterno in [4e5, 6e5, 8e5]: #[4e5, 6e5, 8e5]:
                            self._halve_lr(self.optD)
                            self._halve_lr(self.optG)

                            
                        costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), enc_error.item(), mel_error.item(), rec_error.item()])
                        

                        mean_loss_rec = np.array(costs).mean(0)[-1]
                        mean_loss_mel = np.array(costs).mean(0)[-2]
                        mean_loss_enc = np.array(costs).mean(0)[-3]
                        mean_D_loss = np.array(costs).mean(0)[0]
                        mean_G_loss = np.array(costs).mean(0)[1]
                        info = {
                            f'{self.mode}/D_loss': costs[-1][0],
                            f'{self.mode}/G_loss': costs[-1][1],
                            f'{self.mode}/feature_matching_loss': costs[-1][2],
                            f'{self.mode}/loss_enc': costs[-1][3],
                            f'{self.mode}/loss_mel': costs[-1][4],
                            f'{self.mode}/loss_rec': costs[-1][5],
                            f'{self.mode}/mean_D_loss': mean_D_loss,
                            f'{self.mode}/mean_G_loss': mean_G_loss,
                            f'{self.mode}/mean_loss_enc': mean_loss_enc,
                            f'{self.mode}/mean_loss_mel': mean_loss_mel,
                            f'{self.mode}/mean_loss_rec': mean_loss_rec,
                        }
                        
                        for tag, value in info.items():
                            self.writer.add_scalar(tag, value, total_iterno)
                        
                        train_data_loader.set_description(
                            (
                                f'total_epoch: {accum_epochs}; total_iterno: {accum_iterno}; D_loss: {costs[-1][0]:.3f}; G_loss: {costs[-1][1]:.3f}; mean_loss_enc: {mean_loss_enc:.3f}; mean_loss_mel: {mean_loss_mel:.3f}; mean_loss_rec: {mean_loss_rec:.3f}'
                            )
                        )
                        
                        # Print info.
                        if total_iterno > 0 and total_iterno % self.hps.print_info_every == 0:
                            
                            slot_value = (
                                accum_epochs, 
                                self.hps.vqvae_epochs, 
                                iterno, 
                                len(self.train_data_loader), 
                                accum_iterno
                            ) + tuple([value for value in info.values()][-5:]) + tuple([1000 * (time.time() - start) / self.hps.print_info_every])
                            
                            log = 'MelGAN Stats | Epochs: [%05d/%05d] | Iters: [%05d/%05d] | Total Iters: %d | Mean D Loss: %.3f | Mean G Loss: %.3f | Mean Enc Loss: %.3f | Mean Mel Loss: %.3f | Mean Rec Loss: %.3f | Ms/Batch: %5.2f'
                            #print(log % slot_value)
                            train_data_loader.write(log % slot_value)
                            # Clear costs.
                            costs = []
                            start = time.time()
                    
                    if epoch % self.hps.valid_every_n_epoch == 0:
                        if test_data_loader:
                            train_data_loader.write("Evaluating mean reconstruction loss on test data...")
                            mean_loss_mel = self.evaluate_melgan(test_data_loader)

                        if total_iterno > self.hps.start_save_best_model and mean_loss_mel < loss_mel_best:
                            loss_mel_best = mean_loss_mel
                            is_best_loss = True
                            self.save_model(model_path, self.mode, accum_epochs, iterno, accum_iterno, is_best_loss)
                            #print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{self.total_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                            train_data_loader.write(f"Model saved: {self.mode}, epoch: {accum_epochs}, iterno: {iterno}, total_iterno:{accum_iterno}, loss:{mean_loss_mel}, is_best_loss:{is_best_loss}")
                        else:
                            is_best_loss = False
                            self.save_model(model_path, self.mode, accum_epochs, iterno, accum_iterno, is_best_loss)
                            #print(f"Model saved: {self.mode}, epoch: {epoch}, iterno: {iterno}, total_iterno:{self.total_iterno}, loss:{mean_loss_rec}, is_best_loss:{is_best_loss}")
                            train_data_loader.write(f"Model saved: {self.mode}, epoch: {accum_epochs}, iterno: {iterno}, total_iterno:{accum_iterno}, loss:{mean_loss_mel}, is_best_loss:{is_best_loss}")
                    #train_data_loader.update()

                    if total_iterno >= self.hps.max_iters:
                        train_data_loader.write(f"total_iterno reaches max training iter: {self.hps.max_iters}. Training stop.")
                        break

                    train_data_loader.write("-" * 200)
                
    def evaluate_vqvae(self, test_data_loader):
        costs = []
        self.vqvae.eval()
        with torch.no_grad():
            test_data_loader = tqdm(test_data_loader, total=len(test_data_loader))
            for idx, (x, speaker_info) in enumerate(test_data_loader):
                x = x.to(self.device)
                speaker_id = speaker_info['source_speaker']['id'].to(self.device)
                print(speaker_id)
                print(speaker_info['source_speaker']['name'])
                x_mel = self.audio2mel(x)
                x_rec, _ = self.vqvae(x_mel, speaker_id)
                loss_rec = self.criterion(x_rec, x_mel)
                costs.append(loss_rec.item())
            mean_loss_rec = np.array(costs).mean(0)
        return mean_loss_rec
    def evaluate_melgan(self, test_data_loader):
        return 0.0

if __name__ == "__main__":

    print("Seed: ", seed)
    
    Hps = HyperParams()
    hps = Hps.get_tuple()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--load_vqvae", action='store_true')
    parser.add_argument("--load_melgan", action='store_true')
    parser.add_argument("--pretrained_melgan", type=str, default=None)
    parser.add_argument("--pretrained_vqvae", type=str, default=None)
    parser.add_argument("--specify_speaker", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument("--language", type=str, default="english")
    # mode = "melgan"
    # load_vqvae = False
    # load_melgan = False
    args = parser.parse_args()

    data_path = os.path.join(args.datadir, args.language)

    if not os.path.exists(os.path.join(args.ckpt_path, "logs")):
        os.makedirs(os.path.join(args.ckpt_path, "logs"))
        
    if args.mode == "vqvae":
        print("[VQVAE] Loading training data...")
        rec_train_dataset = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=hps.seg_len, sampling_rate=16000, mode='reconst')
        num_speaker = rec_train_dataset.get_speaker_num()
        speaker2id = rec_train_dataset.get_speaker2id()
        #test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
        train_data_loader = DataLoader(rec_train_dataset, batch_size=hps.batch_size_vqvae, shuffle=True, num_workers=4, pin_memory=True)#hps.batch_size, num_workers=4)
        #test_data_loader = DataLoader(test_set, batch_size=1)
        trainer = Trainer(
            hps=hps, 
            logger_path=os.path.join(args.ckpt_path, "logs"),
            train_data_loader=train_data_loader, 
            mode=args.mode, 
            num_speaker=num_speaker, 
        )

        #print("Loading test data...")
        #test_dataset = AudioDataset(audio_files=Path(data_path) / "eval_files.txt", segment_length=2048 * 126, sampling_rate=16000, mode="reconst", augment=False, speaker2id=speaker2id)
        #test_data_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True)#hps.batch_size, num_workers=4)
    
    elif args.mode == "melgan":
        assert args.load_vqvae == True, "Specify the pre-trained VQ-VAE model --load_vqvae"
        print("[MelGAN] Loading training data...")
        if not args.specify_speaker:
            gan_train_dataset = AudioDataset(audio_files=Path(data_path) / "gan_train_files.txt", segment_length=hps.seg_len, sampling_rate=16000, mode='reconst')
            num_speaker = gan_train_dataset.get_speaker_num()
        else:
            gan_train_dataset = AudioDataset(audio_files=Path(data_path) / "gan_train_files.txt", segment_length=hps.seg_len, sampling_rate=16000, mode='reconst', specify_speaker=args.specify_speaker)
            num_speaker = -1
        #test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
        train_data_loader = DataLoader(gan_train_dataset, batch_size=hps.batch_size_melgan, shuffle=True, num_workers=4, pin_memory=True)#hps.batch_size, num_workers=4)
        #train_data_loader = DataLoader(gan_train_dataset, batch_size=1, shuffle=False, num_workers=4)#hps.batch_size, num_workers=4)
        #test_data_loader = DataLoader(test_set, batch_size=1)
        trainer = Trainer(
            hps=hps,
            logger_path=os.path.join(args.ckpt_path, "logs"), 
            train_data_loader=train_data_loader, 
            mode=args.mode, 
            num_speaker=num_speaker
        )
    

    model_path = os.path.join(args.ckpt_path, "model")
    
    if args.load_vqvae:
        name = 'vqvae' if args.mode == "vqvae" else 'vqvae/encoder_only'
        # vqvae_model = "./ckpts_codebook128_mel64/model-vqvae-10624-0-1582827-True.pt"
        trainer.load_model(args.pretrained_vqvae, name)
    if args.load_melgan:
        name = 'melgan'
        trainer.load_model(args.pretrained_melgan, name)
    trainer.train(save_model_path=model_path)
