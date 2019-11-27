# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 19:01:12
@LastEditors: houwx
@LastEditTime: 2019-11-25 20:44:21
@Description: 
'''

import torch
import json
import h5py
import numpy as np
from collections import namedtuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, h5py_path, index_path, dset="train", seg_len=64, load_mel=False, load_mfcc=False):
        self.dataset = h5py.File(h5py_path, 'r')
        with open(index_path) as f_index:
            self.indexes = json.load(f_index)
        self.indexer = namedtuple('index', ['speaker', 'i', 't'])
        self.seg_len = seg_len
        self.dset = dset
        self.load_mel = load_mel
        self.load_mfcc = load_mfcc
    
    def __getitem__(self, i):
        index = self.indexes[i]
        index = self.indexer(**index)
        speaker_id = index.speaker

        i, t = index.i, index.t # i: speaker / utt_id, t: random start point
        seg_len = self.seg_len
        
        data = [speaker_id, self.dataset[f'{self.dset}/{i}/lin'][t:t+seg_len]]
        if self.load_mel:
            data.append(self.dataset[f'{self.dset}/{i}/mel'][t:t+seg_len])
        if self.load_mfcc:
            data.append(self.dataset[f'{self.dset}/{i}/mfcc'][t:t+seg_len])
        '''
        if self.load_mel:
            data = [speaker_id, self.dataset[f'{self.dset}/{i}/lin'][t:t+seg_len], self.dataset[f'{self.dset}/{i}/mel'][t:t+seg_len]]
        else:
            data = [speaker_id, self.dataset[f'{self.dset}/{i}/lin'][t:t+seg_len]]
        '''
        return tuple(data)
    
    def __len__(self):
        return len(self.indexes)


class DataLoader(object):
    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_elements = len(self.dataset[0]) # Number of elements, 2 to 4: speaker_id, linear spec, (mel_spec, mfcc_spec)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        samples = [self.dataset[self.index + i] for i in range(self.batch_size)]
        #print(len(samples))
        batch = [[s for s in sample] for sample in zip(*samples)]
        #print(len(batch))
        batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]
        #print(len(batch_tensor))
        if self.index + 2 * self.batch_size >= len(self.dataset):
            self.index = 0
        else:
            self.index += self.batch_size
        # print(batch_tensor[1].shape) # lin spec: (batch size, seq len, 1+n_fft//2)
        return tuple(batch_tensor)

if __name__ == "__main__":
    dataset = Dataset(h5py_path="./dataset/english/dataset.hdf5",
            index_path="./dataset/english/index.json",
            load_mel=True,
            load_mfcc=True)
    data_loader = DataLoader(dataset)
    for iteration in range(1):
        data = next(data_loader)
        #print(data)