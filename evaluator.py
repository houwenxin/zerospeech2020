# -*- coding: UTF-8 -*-

from dataset import AudioDataset
from pathlib import Path
import os
from torch.utils.data import DataLoader
from hps import HyperParams
from model.modules import Audio2Mel

import librosa
#from librosa.feature.inverse import mel_to_audio
import numpy as np
import torch
import torch.nn as nn

import soundfile as sf
import copy
import argparse
from model.vqvae.vqvae import VQVAE
from model.melgan.melgan import Generator, Discriminator

class Evaluator(object):
    def __init__(self, hps, encoding_data_loader, convert_data_loader, mode="vqvae", num_speaker=-1, num_src_speaker=-1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using cuda: ", torch.cuda.is_available())

        self.hps = hps # Hyper-Parameters
        self.encoding_data_loader = encoding_data_loader # Use tqdm progress bar
        self.convert_data_loader = convert_data_loader
        self.mode = mode
        
        self.num_speaker = num_speaker
        self.num_src_speaker = num_src_speaker

        self.criterion = nn.L1Loss() # Use L1 Loss
        self.build_model()

    def build_model(self):
        self.audio2mel = Audio2Mel().to(self.device)
        self.vqvae = VQVAE(in_channel=80, channel=512, 
                            embed_dim=self.hps.vqvae_embed_dim, 
                            n_embed=self.hps.vqvae_n_embed,
                            num_speaker=self.num_src_speaker,
                            ).to(self.device)
        if self.mode == 'vqvae':
            return
        elif self.mode in ["melgan", "both"]: # Embeddings from VQVAE: (B, embed_dim, T_mel / 4)
            input_dim = 2 * self.hps.vqvae_embed_dim
            if self.num_speaker != -1:
                self.spk_embed = nn.Embedding(self.num_speaker, self.hps.vqvae_embed_dim // 2).to(self.device)
                input_dim += self.hps.vqvae_embed_dim // 2
            self.netG = Generator(input_dim, self.hps.ngf, self.hps.n_residual_layers).to(self.device)
            self.netD = Discriminator(self.hps.num_D, self.hps.ndf, self.hps.n_layers_D, self.hps.downsamp_factor).to(self.device)
        else:
            raise NotImplementedError("Invalid Mode!")

    def load_model(self, model_file, name):
        print('[Evaluator] - load model from {}'.format(model_file))
        model = torch.load(model_file)
        if name == "vqvae":
            self.vqvae.load_state_dict(model['vqvae'])
        elif name == "vqvae/encoder_only":
            model_dict=self.vqvae.state_dict()
            # Filter out decoder's keys in state dict
            load_dict = {k: v for k, v in model['vqvae'].items() if k in model_dict and "dec" not in k and "spk" not in k}
            model_dict.update(load_dict)
            self.vqvae.load_state_dict(model_dict)
        elif name == "melgan":
            self.netG.load_state_dict(model['generator'])
            self.netD.load_state_dict(model['discriminator'])
            if self.num_speaker != -1:
                self.spk_embed.load_state_dict(model['spk_embed'])
        else:
            raise NotImplementedError("Invalid Model Name!")
        print(f"{name} loaded.")
    
    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def eval_vqvae(self, save_path):
        costs = []
        self.vqvae.eval()
        with torch.no_grad():
            for idx, (x, speaker_info) in enumerate(self.encoding_data_loader):
                x = x.to(self.device)

                source_speaker_name = speaker_info['source_speaker']['name'][0]
                source_speaker_id = speaker_info['source_speaker']['id'].to(self.device)
    
                x_mel = self.audio2mel(x)

                x_rec, _, x_enc = self.vqvae(x_mel, source_speaker_id)

                quant_t, quant_b, _, id_t, id_b = self.vqvae.encode(x_mel)
                x_enc = self.vqvae.get_encoding(quant_t, quant_b).detach()
                x_id = torch.cat((id_t, id_b), dim=1)
                
                x_id = self.to_categorical(x_id.cpu().numpy(), num_classes=128)
                x_id = x_id.squeeze()
                
                save_name = str(source_speaker_name) + '_' + str(speaker_info['speech_id'][0]) + '.txt'
                
                loss_rec = self.criterion(x_rec, x_mel)
                costs.append(loss_rec.item())
                
                np.savetxt(save_path + save_name, x_id, fmt='%d')
                print(save_name)


        print("Mean Rec Loss: ", np.array(costs).mean(0))

    def eval_melgan(self, save_path):
        print("Generating outputs...")
        self.vqvae.eval()
        self.netD.eval()
        self.netG.eval()
        self.spk_embed.eval()
        with torch.no_grad():
            for idx, (x, speaker_info, audio_length) in enumerate(self.convert_data_loader):
                #source_speaker_id = speaker_info['source_speaker']['id']
                source_speaker_name = speaker_info['source_speaker']['name'][0]
                target_speaker_id = speaker_info['target_speaker']['id'].to(self.device)
                target_speaker_name = speaker_info['target_speaker']['name'][0]

                x = x.to(self.device)
                x_mel = self.audio2mel(x).detach()
                quant_t, quant_b, _, _, _ = self.vqvae.encode(x_mel)
                x_enc = self.vqvae.get_encoding(quant_t, quant_b).detach() # x_enc: (B, 2 * embed_dim, T/4)
                
                spk = self.spk_embed(target_speaker_id)
                spk = spk.view(spk.size(0), spk.size(1), 1)  # Return: (B, embed_dim // 2, 1)
                spk_expand = spk.expand(spk.size(0), spk.size(1), x_enc.size(2)) # Return: (B, embed_dim // 2, T / 4)
                #print("\n", spk_expand.shape)
                x_enc = torch.cat((x_enc, spk_expand), dim=1) 
                wav = self.netG(x_enc.to(self.device))

                wav = wav.to('cpu').squeeze()[:audio_length]
                #sf.write(save_path + 'org_' + str(idx) + '_' + str(source_speaker_name) + '.wav', x.to('cpu').squeeze(), 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                #sf.write(save_path + 'melgan_' + str(idx) + '_' + str(target_speaker_name) + '.wav', wav.to('cpu').squeeze(), 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                sf.write(save_path + str(target_speaker_name) + '_' + str(speaker_info['speech_id'][0]) + '.wav', wav, 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
                print(str(target_speaker_name) + '_' + str(speaker_info['speech_id'][0]) + '.wav')
                
    def evaluate(self, save_path):
        self.vqvae.eval()
        if self.mode in ["melgan", 'both']:
            self.netG.eval()
            self.eval_melgan(save_path)
        if self.mode in ['vqvae', 'both']:
            self.eval_vqvae(save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--vqvae_model', type=str)
    parser.add_argument('--melgan_model', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--eval_mode', choices=['both', 'vqvae', 'melgan'], default='melgan')
    args = parser.parse_args()

    Hps = HyperParams()
    hps = Hps.get_tuple()
    language = args.language

    print(f"Language: {language}")
    data_path = os.path.join(args.datadir, language)
    vqvae_model = args.vqvae_model
    melgan_model = args.melgan_model
    save_path = args.save_path

    #os.remove(wav_save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        
    #hps.seg_len = 16000 * 10
    eval_mode = "both"


    if eval_mode in ['vqvae', 'both']:
        encoding_dataset = AudioDataset(audio_files=Path(data_path) / "eval_files.txt", segment_length=2048 * 126, sampling_rate=16000, mode='reconst', augment=False, load_speech_id=True)
        #dataset = AudioDataset(audio_files=Path(data_path) / "eval_files.txt", segment_length=2048 * 126, sampling_rate=16000, mode=data_mode, augment=False, load_speech_id=True)
        num_src_speaker = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=2048 * 126, sampling_rate=16000, mode='reconst', augment=False).get_speaker_num()

    if eval_mode in ['melgan', 'both']:
        convert_dataset = AudioDataset(audio_files=Path(data_path) / "synthesis.txt", segment_length=2048 * 126, sampling_rate=16000, mode='convert', augment=False, load_speech_id=True, return_audio_length=True)
        
    
    num_speaker = 1 if language == 'surprise' else 2 # dataset.get_speaker_num()
    
    encoding_data_loader = DataLoader(encoding_dataset, batch_size=1)#hps.batch_size, num_workers=4)
    convert_data_loader = DataLoader(convert_dataset, batch_size=1)

    evaluator = Evaluator(hps, encoding_data_loader=encoding_data_loader, convert_data_loader=convert_data_loader, mode=eval_mode, num_speaker=num_speaker, num_src_speaker=num_src_speaker)
    
    if eval_mode in ['vqvae', 'both']:
        print(f"Evaluating VQVAE using: {vqvae_model}")
        evaluator.load_model(vqvae_model, 'vqvae')

    if eval_mode in ['melgan', 'both']:
        print(f"Evaluating MelGAN using VQVAE: {vqvae_model} | MelGAN: {melgan_model}")
        # evaluator.load_model(vqvae_model, 'vqvae/encoder_only')
        evaluator.load_model(melgan_model, 'melgan')
    
    evaluator.evaluate(save_path)

    
