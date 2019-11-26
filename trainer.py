# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 20:50:56
@LastEditors: houwx
@LastEditTime: 2019-11-25 20:51:21
@Description: 
'''

from torch import optim

class Trainer(object):
    def __init__(self, hps, data_loader, mode="vqvae"):
        self.hps = hps # Hyper-Parameters
        self.data_loader = data_loader
        self.mode = mode
        self.build_model()
    
    def build_model(self):
        self.encoder_1 = Encoder()
        self.encoder_2 = Encoder()
        self.vqlayer = VQLayer()
        if mode == "vqvae":
            self.decoder = Decoder()
            params = list(self.encoder_1.parameters()) + self.encoder_2.parameters()) + self.vqlayer.parameters()) + self.decoder.parameters())
            self.vqvae_optimizer = optim.Adam(params)
        elif mode == "gan":
            self.generator = Generator()
            self.discriminator = Discriminator()
            self.gen_optimizer = optim.Adam(self.generator.parameters())
            self.dis_optimizer = optim.Adam(self.generator.parameters())
        else:
            raise NotImplementedError("Invalid Mode!")
        
