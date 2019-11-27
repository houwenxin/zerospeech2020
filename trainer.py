# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 20:50:56
@LastEditors: houwx
@LastEditTime: 2019-11-25 20:51:21
@Description: 
'''

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from model.vqvae.vqvae import VQVAE
from model.modules import Audio2Mel

class Trainer(object):
    def __init__(self, hps, data_loader, mode="vqvae"):
        self.hps = hps # Hyper-Parameters
        self.data_loader = data_loader # Mel Input Shape: (B, T, 80)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.audio2mel = Audio2Mel().to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=256).to(self.device)
        if self.mode == "vqvae":
            self.vqvae_optimizer = optim.Adam(self.vqvae.parameters(), lr=self.hps.lr)
        else:
            raise NotImplementedError("Invalid Mode!")
    
    def save_model(self, model_path, name, iteration):
        if name == "vqvae":
            model = self.vqvae.state_dict()

        new_model_path = '{}-{}-{}'.format(model_path, name, iteration)
        torch.save(model, new_model_path)
        #self.model_kept.append(new_model_path)

        #if len(self.model_kept) >= self.max_keep:
        #    os.remove(self.model_kept[0])
        #    self.model_kept.pop(0)
    def load_model(self, model_path):
        model = torch.load(model_path)
        return model

    def train(self, model_path, target_guided=False):
        if self.mode == "vqvae":
            best_rec_loss = 1000000
            for epoch in range(1, self.hps.vqvae_epochs + 1):
                for iterno, x in enumerate(self.data_loader):
                    x = x.to(self.device)
                    x_mel = self.audio2mel(x).detach().to(self.device)
                    x_rec, _ = self.vqvae(x_mel)
                    loss_rec = F.l1_loss(x_rec, x_mel)
                    loss_rec.backward()
                    self.vqvae_optimizer.step()

                    # Print info
                    info = {f'{self.mode}/pre_loss_rec': loss_rec.item()}
                    slot_value = (epoch, self.hps.vqvae_epochs, iterno) + tuple([value for value in info.values()])
                    log = 'VQVAE:[%06d/%06d], iter=%d, loss_rec=%.3f'
                    print(log % slot_value)
            #self.save_model(model_path, self.mode, self.hps.vqvae_pretrain_iters)

if __name__ == "__main__":
    from dataset import AudioDataset
    from pathlib import Path
    from torch.utils.data import DataLoader
    from hps.hps import Hps
    Hps = Hps()
    hps = Hps.get_tuple()
    data_path = "./databases/english_small/"
    rec_train_dataset = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=hps.seg_len, sampling_rate=22050)
    #test_set = AudioDataset(audio_files=Path(data_path) / "test_files.txt", segment_length=22050 * 4, sampling_rate=22050, augment=False)
    train_data_loader = DataLoader(rec_train_dataset, batch_size=hps.batch_size, num_workers=4)
    #test_data_loader = DataLoader(test_set, batch_size=1)
    trainer = Trainer(hps=hps, data_loader=train_data_loader, mode="vqvae")

    trainer.train("./ckpts/")
