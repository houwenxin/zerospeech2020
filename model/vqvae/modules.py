# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2020-01-20 15:04:36
@LastEditors  : houwx
@LastEditTime : 2020-01-20 15:05:29
@Description: 
'''
import torch.nn as nn
from model.modules import weights_init

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(channel),
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out