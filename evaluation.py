# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-20 13:39:02
@LastEditors: houwx
@LastEditTime: 2019-12-11 16:19:56
@Description: 
'''
from dataset import AudioDataset
from pathlib import Path
import os
from torch.utils.data import DataLoader
from hps.hps import HyperParams
from model.modules import Audio2Mel

import librosa
#from librosa.feature.inverse import mel_to_audio
import numpy as np
import torch
import torch.nn as nn

import soundfile as sf
import copy

from model.vqvae.vqvae import VQVAE

class Mel2Audio(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def griffin_lim(self, mag): # Applies Griffin-Lim's law.
        def _invert_spectrogram(spectrogram): # spectrogram: [f, t]
            return librosa.istft(spectrogram, self.hop_length, win_length=self.win_length, window="hann", center=False)
        n_iter = 300
        X_best = copy.deepcopy(mag)
        for _ in range(n_iter):
            X_t = _invert_spectrogram(X_best)
            est = librosa.stft(X_t, self.n_fft, self.hop_length, win_length=self.win_length, center=False)
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = mag * phase
        X_t = _invert_spectrogram(X_best)
        y = np.real(X_t)
        return y
        
    def forward(self, mel):
        mel = np.power(10.0, mel)
        
        def _build_mel_basis():
            return librosa.filters.mel(self.sampling_rate, self.n_fft, n_mels=self.n_mel_channels)
        inv_mel_basis = np.linalg.pinv(_build_mel_basis())
        mag = np.dot(inv_mel_basis, mel)
        wav = self.griffin_lim(mag)
        #wav = librosa.feature.inverse.mel_to_audio(mel, n_iter=500, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length, center=False, win_length=self.win_length, power=2)
        return wav
        
if __name__ == "__main__":
    Hps = HyperParams()
    hps = Hps.get_tuple()
    data_path = "../../databases/english/" # On lab server.
    #data_path = "./databases/english_small/" # On my own PC.
    #hps.seg_len = 16000 * 10
    rec_train_dataset = AudioDataset(audio_files=Path(data_path) / "rec_train_files.txt", segment_length=16000 * 4, sampling_rate=16000)
    train_data_loader = DataLoader(rec_train_dataset, batch_size=1)#hps.batch_size, num_workers=4)
    num_speaker = rec_train_dataset.get_speaker_num()
    audio2mel = Audio2Mel()
    mel2audio = Mel2Audio()
    
    vqvae = VQVAE(in_channel=80, channel=512, 
                    embed_dim=hps.vqvae_embed_dim, 
                    n_embed=hps.vqvae_n_embed,
                    add_speaker_id=True,
                    num_speaker=num_speaker
                )
    model = torch.load("model-vqvae-14058-0-2094493-True.pt", map_location='cpu')
    vqvae.load_state_dict(model['vqvae'])
    wav_path = "./recon_wavs/"
    for idx, (x, speaker_id) in enumerate(train_data_loader):
        vqvae.eval()
        #print(idx)
        x_mel = audio2mel(x)
        x_rec, _ = vqvae(x_mel, speaker_id)
        x_mel = np.array(x_mel).squeeze(0)
        x_rec = np.array(x_rec.detach()).squeeze(0)
        #print(x_mel.shape)
        wav = mel2audio(x_mel)
        vqvae_wav = mel2audio(x_rec)
        print(x.squeeze())
        print(wav)
        print(vqvae_wav)
        print('------------------------------------------------------' * 2)
        speaker_id = int(speaker_id)
        sf.write(wav_path + 'org_' + str(idx) + '_' + str(speaker_id) + '.wav', x.squeeze(), 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
        sf.write(wav_path + 'rec_' + str(idx) + '_' + str(speaker_id) + '.wav', wav, 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
        sf.write(wav_path + 'vqvae_' + str(idx) + '_' + str(speaker_id) + '.wav', vqvae_wav, 16000, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
        if idx == 10:
            break
	    #sf.write(wav_path2, wav_data_mel, hp.sr, 'PCM_16')
        
