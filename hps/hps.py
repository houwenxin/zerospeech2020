# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-18 15:38:23
@LastEditors: houwx
@LastEditTime: 2019-11-18 16:17:05
@Description: Hyper-Parameters
'''

class preprocessing_hyperparams(object):
    '''
     @description: Hyper-Parametrs for Preprocessing
     @param {type} 
     @return: 
     '''
    def __init__(self):
        self.max_duration = 10.0

        # signal processing
        self.sr = 16000 # Sample rate.
        self.n_fft = 1024 # fft points (samples)
        self.frame_shift = 0.0125 # seconds
        self.frame_length = 0.05 # seconds
        self.hop_length = int(self.sr*self.frame_shift) # samples.
        self.win_length = int(self.sr*self.frame_length) # samples.
        self.n_mels = 80 # Number of Mel banks to generate
        self.power = 1.2 # Exponent for amplifying the predicted magnitude
        self.n_iter = 300 # Number of inversion iterations
        self.preemphasis = .97 # or None
        self.max_db = 100
        self.ref_db = 20
        self.prior_freq = 3000
        self.prior_weight = 0.5
        
hp = preprocessing_hyperparams()