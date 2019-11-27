# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-20 13:39:02
@LastEditors: houwx
@LastEditTime: 2019-11-25 15:35:47
@Description: 
'''
import librosa
import scipy
import numpy as np
from scipy.fftpack import dct
import copy


def load_wav(path, sr=16000):
    return librosa.core.load(path, sr=sr)[0] # Load an audio file as a floating point time series.


#==================================== Preprocess ===========================================
def preemphasis(x, preemphasis=0.97):
    #np.append(x[0], x[1:] - preemphasis * x[0:-1]) # 预加重：y(n)=x(n)-ax(n-1) 其中a为预加重系数，一般是0.9~1.0之间，通常取0.98。这里取0.97
    return scipy.signal.lfilter([1, -preemphasis], [1], x) # np.float64, More Accurate


def trim_silence(wav):
    # Trim leading and trailing silence from an audio signal.
	return librosa.effects.trim(wav)[0]


def rescale(wav, hp):
    if hp.rescale:
        wav = wav / np.abs(wav).max() * hp.rescaling_max
    return wav


def linear_spectrogram(y, hp):
    fft_windows = librosa.stft(y=y, 
                        n_fft=hp.n_fft, 
                        hop_length=hp.hop_length, 
                        win_length=hp.win_length,
                        center=False,
                        ) # Short-time Fourier transform (STFT)
    magnitude = np.abs(fft_windows)  # magnitude spectrogram: (1+n_fft//2, T)
    return magnitude

def linear2mel(linear_spectrogram, hp):
    mel_filter = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels, fmin=0.0, fmax=None) # Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
    mel = np.dot(mel_filter, linear_spectrogram)
    return mel


def mel2mfcc(mel, hp):
    dct_type=2
    norm='ortho'
    mfcc = dct(mel, axis=0, type=dct_type, norm=norm)[:hp.n_mfcc]
    mfcc_delta = _delta(mfcc)
    mfcc_delta_delta = _delta(mfcc_delta)
    mfccs = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))
    return mfccs


def _delta(S):
    delta = librosa.feature.delta(S, width=7)
    return delta


def amp2db(amp):
    return 20 * np.log10(np.maximum(1e-5, amp))


def normalize(y, hp):
    return np.clip((y - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)


# ======================================== Convert =====================================
def griffin_lim(spectrogram, hp): # Applies Griffin-Lim's law.
	
    def _invert_spectrogram(spectrogram): # spectrogram: [f, t]
        return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann", center=False)
    
    X_best = copy.deepcopy(spectrogram)
    for _ in range(hp.n_iter):
        X_t = _invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length, center=False)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = _invert_spectrogram(X_best)
    y = np.real(X_t)
    return y


def linear2wav(mag, hp): # Generate wave file from spectrogram
	mag = mag.T # transpose
	mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db # de-normalize
	mag = np.power(10.0, mag * 0.05) # to amplitude
	wav = griffin_lim(mag, hp) # wav reconstruction
	wav = scipy.signal.lfilter([1], [1, -hp.preemphasis], wav) # de-preemphasis
	wav, _ = librosa.effects.trim(wav) # trim
	return wav.astype(np.float32)


def mel2linear(mel_spectrogram, hp):
    def _build_mel_basis():
        return librosa.filters.mel(hp.sr, hp.n_fft, n_mels=hp.n_mels)
    inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.dot(inv_mel_basis, mel_spectrogram)


def mel2wav(mel, hp):
    mel = mel.T
    mel = (np.clip(mel, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db # de-noramlize
    mel = np.power(10.0, mel * 0.05) # to amplitude
    mag = mel2linear(mel, hp)
    wav = griffin_lim(mag, hp) # wav reconstruction
    wav = scipy.signal.lfilter([1], [1, -hp.preemphasis], wav) # de-preemphasis
    wav, _ = librosa.effects.trim(wav) # trim
    return wav