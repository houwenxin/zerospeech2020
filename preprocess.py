# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-18 15:15:56
@LastEditors: houwx
@LastEditTime: 2019-11-18 18:05:01
@Description: 
'''
import librosa
import os
import numpy as np
from hps.hps import hp

src_path = "../../databases/english_small/train/unit"
trg_path = "./"

def get_spectrograms(sound_file):
    y, _ = librosa.load(sound_file, sr=hp.sr) # Load an audio file as a floating point time series.
    #print(len(y))
    y, _ = librosa.effects.trim(y) # Trim leading and trailing silence from an audio signal.
    #print(y, ind)
    y = np.append(y[0], y[1:] - hp.preemphasis * y[0:-1]) # 预加重：y(n)=x(n)-ax(n-1) 其中a为预加重系数，一般是0.9~1.0之间，通常取0.98。这里取0.97
    #y = librosa.effects.preemphasis(y, coef=hp.preemphasis) # ??? Why different? Pre-emphasize an audio signal with a first-order auto-regressive filter:
    #print(y)
    fft_windows = librosa.stft(y=y, 
                        n_fft=hp.n_fft, 
                        hop_length=hp.hop_length, 
                        win_length=hp.win_length
                        ) # Short-time Fourier transform (STFT)
    #print(fft_windows.shape)
    magnitude = np.abs(fft_windows)  # magnitude spectrogram: (1+n_fft//2, T)
    #print(magnitude.shape)
    mel_filter = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels) # Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
    mel = np.dot(mel_filter, magnitude)
    #print(mel)
    '''
    S = librosa.feature.melspectrogram(y=y, sr=hp.sr, n_fft=hp.n_fft, 
                                   hop_length=hp.hop_length, 
                                   win_length=hp.win_length,
                                   n_mels=hp.n_mels,
                                   power=1) # Equal when power == 1
    print(S)
    assert (mel == S).all()
    '''
    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, magnitude))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

	# Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag
    



def preprocess(src_path, trg_path):
    pass


if __name__ == "__main__":
    file = os.path.join(src_path, "S133_00324.wav")
    get_spectrograms(file)