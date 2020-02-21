# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-12-06 16:09:06
@LastEditors  : houwx
@LastEditTime : 2020-01-27 13:39:56
@Description: 
'''
import torch.nn as nn
import torch.nn.functional as F
from model.modules import weights_init

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channel),
        )
    def forward(self, input):
        return input + self.conv(input)

class SpeakerClassifier(nn.Module):
    def __init__(self, in_channel, n_class, channel=256, n_res=1, dropout=False):
        super(SpeakerClassifier, self).__init__()
        self.in_channel = in_channel
        self.n_class = n_class
        self.channel = channel
        self.n_res = n_res
        self.dropout = dropout
        self._build_model()
        self.apply(weights_init)

    def _build_model(self):
        blocks = [
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(self.channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(self.channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.channel),
        ]
        for _ in range(self.n_res):
            blocks.append(ResBlock(self.channel, self.channel))
        blocks.append(nn.ReLU(inplace=True)) # B, 64, T
        self.blocks = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.channel, self.n_class) # B, 7, 1
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=0.5)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        out = self.avg_pool(self.blocks(input)).squeeze(2)
        out = self.fc(out)
        if self.dropout:
            out = self.dropout_layer(out)
        out = self.log_softmax(out)
        return out
