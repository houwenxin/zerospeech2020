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

from model.vqvae.vqvae import VQVAE

class Trainer(object):
    def __init__(self, hps, data_loader, mode="vqvae"):
        self.hps = hps # Hyper-Parameters
        self.data_loader = data_loader # Mel Input Shape: (B, T, 80)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
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

    def train(self, model_path, target_guided=False):
        if self.mode == "vqvae":
            for iteration in range(self.hps.vqvae_pretrain_iters):
                speaker_id, _, X = next(self.data_loader)
                X = Variable(X.permute(0, 2, 1), requires_grad=True).to(self.device)
                X_rec, diff = self.vqvae(X)
                print(X_rec.shape)
                print(X.shape)
                loss_rec = torch.mean(torch.abs(X_rec - X))
                loss_rec.backward()
                self.vqvae_optimizer.step()

                # tb info
                info = {f'{self.mode}/pre_loss_rec': loss_rec.item()}
                slot_value = (iteration + 1, self.hps.vqvae_pretrain_iters) + tuple([value for value in info.values()])
                log = 'pre_AE:[%06d/%06d], loss_rec=%.3f'
                #print(log % slot_value)
            self.save_model(model_path, self.mode, self.hps.vqvae_pretrain_iters)

if __name__ == "__main__":
    from dataloader import DataLoader, Dataset
    from hps.hps import Hps
    Hps = Hps()
    hps = Hps.get_tuple()
    dataset = Dataset(h5py_path="./dataset/english/dataset.hdf5",
            index_path="./dataset/english/index.json",
            load_mel=True,
            load_mfcc=False)
    data_loader = DataLoader(dataset, batch_size=16)
    trainer = Trainer(hps=hps, data_loader=data_loader)
    trainer.train("./ckpts/")
