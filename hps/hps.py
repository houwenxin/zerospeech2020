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
        self.n_mfcc = 13 # Number of MFCC to generate
        self.n_iter = 300 # Number of inversion iterations
        #self.power = 1.2 # Exponent for amplifying the predicted magnitude
        
        self.preemphasis = 0.97 # or None
        self.max_db = 100  
        self.ref_db = 20    
        self.prior_freq = 3000
        self.prior_weight = 0.5
        '''
        self.rescale = True
        self.rescaling_max = 0.999
        self.n_mfcc = 13
        self.signal_normalization = True # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        self.symmetric_mels = False # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
        self.min_level_db = -100
        self.ref_level_db = 20
        self.max_abs_value = 4. # Max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,not too small for fast convergence)
        # TODO Fmin Fmax used for _build_mel_basis
        self.fmin = 55 # 55, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
        self.fmax = 7600 # To be increased/reduced depending on data.
        self.magnitude_power = 1
        # Griffin Lim
        self.power = 1.2 #Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
        self.griffin_lim_iters = 60 #Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
        
        '''
hp = preprocessing_hyperparams()