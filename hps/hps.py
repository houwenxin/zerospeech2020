# -*- coding: UTF-8 -*-

import json
from collections import namedtuple

class HyperParams(object):
	def __init__(self, path=None):
		self.hps = namedtuple('hps', [
            'lr_vqvae', # Learning rates
            'lr_melgan',
            'lr_melgan_pretrain',
            'batch_size_vqvae', # Batch size of training data.
            'batch_size_melgan',
            'max_grad_norm', # 5
            
            'seg_len', # Length of segment

            'max_saved_model', # Max number of saved models.
            'max_best_model', # Max number of saved best-loss models.
            'print_info_every', # Print training info every {} iterations.
            'valid_every_n_epoch', # Run validation during training every {} iterations.
            'save_model_every', # Save model during training every {} iterations.
            'start_save_best_model', # Start save best model afer {} iterations.
            
            # ============== VQVAE =============
            'vqvae_epochs',
            'vqvae_n_embed',
            'vqvae_embed_dim',
            'loss_latent_weight', # Weight of commitment loss: 0.25 in paper VQVAE-2
            "max_loss_spk_clf_weight",

            # ============== MelGAN =============
            'melgan_epochs',
            
            # ============== MelGAN Generator =============
            'ngf',
            'n_residual_layers',
            'lambda_feat',

            # ============== MelGAN Discriminator =============
            'num_D', # Number of discriminators
            'ndf',
            'n_layers_D',
            'downsamp_factor',

            "hop_length",


            # =============== Improvement ===================
            'adv_loss_weight',
            'enc_error_weight',
            'mel_error_weight',
            'max_iters',
			]
		)
		if not path is None:
			self.load(path)
			print('[HPS Loader] - Loading from: ', path)
		else:
			print('[HPS Loader] - Using default parameters since no .json file is provided.')
			default = {
                'lr_vqvae':4e-4,
                'lr_melgan':2e-4,
                'batch_size_vqvae':64, # Batch size of training data.
                'batch_size_melgan':16, # Batch size is very IMPORTANT for melgan. Set to 64 will lead to NaN loss quickly.
                'max_grad_norm':5, # 5
                'seg_len':32768, # Segment length loaded from raw wav. default in melgan:8192

                'max_saved_model':3, # Max number of saved models.
                'max_best_model':2, # Max number of saved best-loss models.
                'print_info_every':221,# 149 for VQVAE, 221/442 for MelGAN. # Print training info every {} iterations.
                'save_model_every':221, # 149 for VQVAE, 221/442 for MelGAN. Save model during training every {} iterations.
                'valid_every_n_epoch':1, # Run validation during training every {} epochs.
                'start_save_best_model':0, # Start save best model afer {} iterations.

                # ============== VQVAE =============
                'vqvae_epochs':10000,
                'vqvae_n_embed':128, #512, # Try 256, 128
                'vqvae_embed_dim':64, 
                'loss_latent_weight':0.25, # Weight of commitment loss: 0.25 in paper VQVAE-2
                
                # ============== MelGAN =============
                'melgan_epochs':3000,
                 
                # ============== MelGAN Generator =============
                'ngf':32, # ngf is a model hyperparameter, meaning the final number of feature maps in Generator, 32 in paper.
                'n_residual_layers':3,
                

                # ============== MelGAN Discriminator =============
                'num_D':3, # Number of discriminators
                'ndf':16,
                'n_layers_D':4,
                'downsamp_factor':4,

                "hop_length":256,


                # =============== Auxillary ===================
                "max_loss_spk_clf_weight": 0.05,
                'adv_loss_weight':1, # 2e-1, # 5e-2
                'lambda_feat':10, # 5e-1,
                'enc_error_weight':0.0, #1,
                'mel_error_weight':0.0, #5e-1,
                'max_iters':10e5,
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
