# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-18 15:38:23
@LastEditors: houwx
@LastEditTime: 2019-12-01 18:04:22
@Description: Hyper-Parameters
'''

import json
from collections import namedtuple

class ProcessingHyperParams(object):
    '''
     @description: Hyper-Parametrs for Preprocessing
     @param {type} 
     @return: 
     '''
    def __init__(self):
        self.max_duration = 10.0

        # signal processing
        self.sr = 22050 # 16000 # Sample rate.
        self.n_fft = 1024 # fft points (samples)
        #self.frame_shift = 0.0125 # seconds
        #self.frame_length = 0.05 # seconds
        self.hop_length = 256 #int(self.sr*self.frame_shift) # samples. 200
        self.win_length = 1024 #int(self.sr*self.frame_length) # samples. 800
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
hp = ProcessingHyperParams()


class HyperParams(object):
	def __init__(self, path=None):
		self.hps = namedtuple('hps', [
            'lr', # Learning rate
            'batch_size', # Batch size of training data.
            'max_grad_norm', # 5
            
            'seg_len', # Length of segment

            'max_saved_model', # Max number of saved models.
            'print_info_every', # Print training info every {} iterations.
            'run_valid_every', # Run validation during training every {} iterations.
            'save_model_every', # Save model during training every {} iterations.
            'start_save_best_model', # Start save best model afer {} iterations.
            
            # ============== VQVAE =============
            'vqvae_epochs',
            'vqvae_n_embed',
            'vqvae_embed_dim',

            # ============== MelGAN Generator =============
            'ngf',
            'n_residual_layers',

            # ============== MelGAN Discriminator =============
            'num_D', # Number of discriminators
            'ndf',
            'n_layers_D',
            'downsamp_factor',
			]
		)
		if not path is None:
			self.load(path)
			print('[HPS Loader] - Loading from: ', path)
		else:
			print('[HPS Loader] - Using default parameters since no .json file is provided.')
			default = {
                'lr':1e-4,
                'batch_size':16, # Batch size of training data.
                'max_grad_norm':5, # 5
                'seg_len':8192, # Segment length loaded from raw wav.

                'max_saved_model':8, # Max number of saved models.
                'print_info_every':300, # Print training info every {} iterations.
                'run_valid_every':10, # Run validation during training every {} iterations.
                'save_model_every':3000, # Save model during training every {} iterations.
                'start_save_best_model':1000, # Start save best model afer {} iterations.

                # ============== VQVAE =============
                'vqvae_epochs':3000,
                'vqvae_n_embed':512, # Try 256, 128
                'vqvae_embed_dim':64,

                # ============== MelGAN Generator =============

            }
			self._hps = self.hps(**default)

	def get_tuple(self):
		return self._hps

	def load(self, path):
		with open(path, 'r') as f_json:
			hps_dict = json.load(f_json)
		self._hps = self.hps(**hps_dict)

	def dump(self, path):
		with open(path, 'w') as f_json:
			json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))
