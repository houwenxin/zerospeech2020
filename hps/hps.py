# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-18 15:38:23
@LastEditors: houwx
@LastEditTime: 2019-11-21 14:17:35
@Description: Hyper-Parameters
'''
from collections import namedtuple

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


class Hps(object):
	def __init__(self, path=None):
		self.hps = namedtuple('hps', [
			'g_mode',
			'enc_mode',
			'load_model_list',
			'lr',
			'alpha_dis',
			'alpha_enc',
			'beta_dis', 
			'beta_gen', 
			'beta_clf',
			'lambda_',
			'ns', 
			'enc_dp', 
			'dis_dp', 
			'max_grad_norm',
			'max_step',
			'seg_len',
			'n_samples',
			'enc_size',
			'emb_size',
			'n_speakers',
			'n_target_speakers',
			'n_latent_steps',
			'n_patch_steps', 
			'batch_size',
			'lat_sched_iters',
			'vqvae_pretrain_iters',
			'dis_pretrain_iters', 
			'iters',
			'max_to_keep',
			]
		)
		if not path is None:
			self.load(path)
			print('[HPS Loader] - Loading from: ', path)
		else:
			print('[HPS Loader] - Using default parameters since no .json file is provided.')
			default = \
				['enhanced', 'continues', 1e-4, 1e-5, 1e-4, 0, 0, 0, 10, 0.01, 0.5, 0.1, 5, 5, 128, 400000, 1024, 1024, 102, 2, 5, 0, 32, 50000, 5000, 500, 30000, 60000, 10]
			self._hps = self.hps._make(default)

	def get_tuple(self):
		return self._hps

	def load(self, path):
		with open(path, 'r') as f_json:
			hps_dict = json.load(f_json)
		self._hps = self.hps(**hps_dict)

	def dump(self, path):
		with open(path, 'w') as f_json:
			json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))
