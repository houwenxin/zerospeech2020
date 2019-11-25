# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 20:50:56
@LastEditors: houwx
@LastEditTime: 2019-11-25 20:51:21
@Description: 
'''

class Trainer(object):
    def __init__(self, hps, data_loader):
        self.hps = hps # Hyper-Parameters
        self.data_loader = data_loader

        self.build_model()
    
    def build_model(self):
        pass