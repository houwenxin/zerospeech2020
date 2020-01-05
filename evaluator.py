# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-20 13:39:02
@LastEditors  : houwx
@LastEditTime : 2020-01-05 17:26:40
@Description: 
'''
from dataset import AudioDataset
from pathlib import Path
import os
from torch.utils.data import DataLoader
from hps.hps import HyperParams
from model.modules import Audio2Mel, Mel2Audio

import librosa
#from librosa.feature.inverse import mel_to_audio
import numpy as np
import torch
import torch.nn as nn

import soundfile as sf
import copy

from model.vqvae.vqvae import VQVAE
from model.melgan.melgan import Generator, Discriminator

class Evaluator(object):
    def __init__(self, hps, test_data_loader, mode="vqvae", num_speaker=-1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using cuda: ", torch.cuda.is_available())

        self.hps = hps # Hyper-Parameters
        self.test_data_loader = test_data_loader # Use tqdm progress bar
        self.mode = mode
        
        self.num_speaker = num_speaker

        self.criterion = nn.MSELoss() # Use MSE Loss
        self.build_model()

    def build_model(self):
        self.audio2mel = Audio2Mel().to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=512, 
                            embed_dim=self.hps.vqvae_embed_dim, 
                            n_embed=self.hps.vqvae_n_embed,
                            num_speaker=self.num_speaker,
                            ).to(self.device)
        if self.mode == "vqvae":
            self.mel2audio = Mel2Audio().to(self.device)
        elif self.mode == "melgan": # Embeddings from VQVAE: (B, embed_dim, T_mel / 4)
            input_dim = 2 * self.hps.vqvae_embed_dim
            if self.num_speaker != -1:
                self.spk_embed = nn.Embedding(self.num_speaker, self.hps.vqvae_embed_dim // 2).to(self.device)
                input_dim += self.hps.vqvae_embed_dim // 2
            self.netG = Generator(input_dim, self.hps.ngf, self.hps.n_residual_layers).to(self.device)
            self.netD = Discriminator(self.hps.num_D, self.hps.ndf, self.hps.n_layers_D, self.hps.downsamp_factor).to(self.device)
        else:
            raise NotImplementedError("Invalid Mode!")

    def load_model(self, model_file, name):
        print('[Evaluator] - load model from {}'.format(model_file))
        model = torch.load(model_file)
        if name == "vqvae":
            self.vqvae.load_state_dict(model['vqvae'])
        elif name == "vqvae/encoder_only":
            model_dict=self.vqvae.state_dict()
            # Filter out decoder's keys in state dict
            load_dict = {k: v for k, v in model['vqvae'].items() if k in model_dict and "dec" not in k and "spk" not in k}
            model_dict.update(load_dict)
            self.vqvae.load_state_dict(model_dict)
        elif name == "melgan":
            self.netG.load_state_dict(model['generator'])
            self.netD.load_state_dict(model['discriminator'])
            if self.num_speaker != -1:
                self.spk_embed.load_state_dict(model['spk_embed'])
        else:
            raise NotImplementedError("Invalid Model Name!")
        print(f"{name} loaded.")
    
    def eval_vqvae(self, save_path):
        costs = []
        with torch.no_grad():
            for idx, (x, speaker_info) in enumerate(self.test_data_loader):
                x = x.to(self.device)

                speaker_name = speaker_info['source_speaker']['name'][0]
                speaker_id = speaker_info['source_speaker']['id'].to(self.device)

                x_mel = self.audio2mel(x)
                x_rec, _ = self.vqvae(x_mel, speaker_id)
                loss_rec = self.criterion(x_rec, x_mel)

                costs.append(loss_rec.item())

                # Convert to audio
                wav = self.mel2audio(x_mel.squeeze().to('cpu'))
                vqvae_wav = self.mel2audio(x_rec.squeeze().to('cpu'))
                print(x.squeeze())
                print(wav)
                print(vqvae_wav)
                print('------------------------------------------------------' * 2)
                #speaker_id = int(speaker_id)
                sf.write(save_path + 'org_' + str(idx) + '_' + str(speaker_name) + '.wav', x.to('cpu').squeeze(), 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                sf.write(save_path + 'rec_' + str(idx) + '_' + str(speaker_name) + '.wav', wav, 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                sf.write(save_path + 'vqvae_' + str(idx) + '_' + str(speaker_name) + '.wav', vqvae_wav, 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                if idx == 10:
                    break
            mean_loss_rec = np.array(costs).mean(0)
        print("Mean Rec Loss: ", mean_loss_rec)

    def eval_melgan(self, save_path):
        with torch.no_grad():
            for idx, (x, speaker_info) in enumerate(self.test_data_loader):
                #source_speaker_id = speaker_info['source_speaker']['id']
                source_speaker_name = speaker_info['source_speaker']['name'][0]
                target_speaker_id = speaker_info['target_speaker']['id'].to(self.device)
                target_speaker_name = speaker_info['target_speaker']['name'][0]

                x = x.to(self.device)
                x_mel = self.audio2mel(x).detach()
                quant_t, quant_b, _, _, _ = self.vqvae.encode(x_mel)
                x_enc = self.vqvae.get_encoding(quant_t, quant_b).detach() # x_enc: (B, 2 * embed_dim, T/4)
                
                spk = self.spk_embed(target_speaker_id)
                spk = spk.view(spk.size(0), spk.size(1), 1)  # Return: (B, embed_dim // 2, 1)
                spk_expand = spk.expand(spk.size(0), spk.size(1), x_enc.size(2)) # Return: (B, embed_dim // 2, T / 4)
                #print("\n", spk_expand.shape)
                x_enc = torch.cat((x_enc, spk_expand), dim=1) 
                wav = self.netG(x_enc.to(self.device))

                sf.write(save_path + 'org_' + str(idx) + '_' + str(source_speaker_name) + '.wav', x.to('cpu').squeeze(), 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                sf.write(save_path + 'melgan_' + str(idx) + '_' + str(target_speaker_name) + '.wav', wav.to('cpu').squeeze(), 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
    
    def evaluate(self, save_path):
        self.vqvae.eval()
        if self.mode == "melgan":
            self.netG.eval()
            self.eval_melgan(save_path)
        else:
            self.eval_vqvae(save_path)
        

if __name__ == "__main__":
    Hps = HyperParams()
    hps = Hps.get_tuple()
    data_path = "../../databases/english/" # On lab server.
    #data_path = "./databases/english_small/" # On my own PC.
    
    vqvae_model = "./ckpts/model-vqvae-5998-0-893553-True.pt"
    melgan_model = "./ckpts/model-melgan-6760-0-378504-True.pt"
    wav_save_path = "./recon_wavs/"
    #hps.seg_len = 16000 * 10
    eval_mode = "melgan"

    if eval_mode == 'vqvae':
        data_mode = 'reconst'
    elif eval_mode == 'melgan':
        data_mode = 'convert'
    
    dataset = AudioDataset(audio_files=Path(data_path) / "synthesis.txt", segment_length=2048 * 40, sampling_rate=16000, mode=data_mode, augment=False)
    data_loader = DataLoader(dataset, batch_size=1)#hps.batch_size, num_workers=4)
    
    if eval_mode == 'vqvae':
        num_speaker = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=2048 * 40, sampling_rate=16000).get_speaker_num()
    elif eval_mode == 'melgan':
        num_speaker = dataset.get_speaker_num()
    
    evaluator = Evaluator(hps, data_loader, mode=eval_mode, num_speaker=num_speaker)
    
    if eval_mode == 'vqvae':
        evaluator.load_model(vqvae_model, 'vqvae')
    else:
        evaluator.load_model(vqvae_model, 'vqvae/encoder_only')
        evaluator.load_model(melgan_model, 'melgan')
    evaluator.evaluate(wav_save_path)

    
