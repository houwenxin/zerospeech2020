# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-12-03 16:55:41
@LastEditors: houwx
@LastEditTime: 2019-12-03 17:49:48
@Description: 
'''
import torch
from model.vqvae.vqvae import VQVAE
from model.modules import Audio2Mel
from model.melgan.melgan import Generator, Discriminator

class Predictor(object):
    def __init__(self, hps, test_data_loader):
        self.hps = hps
        self.test_data_loader = test_data_loader # Test loader's batch size should be 1.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using cuda: ", torch.cuda.is_available())

        self.audio2mel = Audio2Mel().to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=256, embed_dim=self.hps.vqvae_embed_dim, n_embed=self.hps.vqvae_n_embed).to(self.device)
        self.netG = Generator(self.hps.vqvae_n_embed, self.hps.ngf, self.hps.n_residual_layers).to(self.device)
        self.netD = Discriminator(self.hps.num_D, self.hps.ndf, self.hps.n_layers_D, self.hps.downsamp_factor)
    
    def load_model(self, vqvae_model, melgan_model=None):
        print('[Predictor] - load VQVAE model from {} - load MelGAN model from {}'.format(vqvae_model, melgan_model))
        vqvae_model = torch.load(vqvae_model)
        self.vqvae = self.vqvae.load_state_dict(vqvae_model['vqvae'])
        self.vqvae.eval()
        if melgan_model:
            melgan_model = torch.load(melgan_model)
            self.netG.load_state_dict(melgan_model['generator'])
            self.netD.load_state_dict(melgan_model['discriminator'])
            self.netG.eval()
            self.netD.eval()
        print(f"Loaded.")

    def encode(self, enc_save_path):
        for iterno, (x, speaker_id) in enumerate(self.test_data_loader):
            x = x.to(self.device)
            x_enc = self.vqvae.encode(x).squeeze()
            print(x_enc)
            return
            
    
    def predict(self, wav_save_path):
        pass

if __name__ == "__main__":
    from dataset import AudioDataset
    from pathlib import Path
    import os
    from torch.utils.data import DataLoader
    from hps.hps import HyperParams
    Hps = HyperParams()
    hps = Hps.get_tuple()

    data_path = "../../databases/"
    dataset_name = "english"
    data_path = os.path.join(data_path, dataset_name)
    test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
    test_data_loader = DataLoader(test_set, batch_size=1)
    predictor = Predictor(hps, test_data_loader)
    predictor.encode(dataset_name)