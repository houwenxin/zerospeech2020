import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

import numpy as np
import copy

def weights_init(m):
    classname = m.__class__.__name__
    '''
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    '''
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
        
class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=16000,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        '''
        default hop_length: 256
        '''
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa.filters.mel(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8) # Add a small number (1e-8) to avoid NaN in gradients. 
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

class Mel2Audio(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=64,
        win_length=1024,
        sampling_rate=16000,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        '''
        default hop_length: 256
        '''
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