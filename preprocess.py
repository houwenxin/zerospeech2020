# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-18 15:15:56
@LastEditors: houwx
@LastEditTime: 2019-11-19 20:13:27
@Description: 
'''
import os
import numpy as np
from hps.hps import hp

from utils.convert import spectrogram2wav
import soundfile as sf
from utils import audio

src_path = "../../databases/english_small/train/unit"
trg_path = "./"

def get_spectrograms(sound_file):
    y = audio.load_wav(sound_file, sr=hp.sr)
    print(len(y))
    y = audio.trim_silence(y)
    print(len(y))
    y = audio.preemphasis(y, hp.preemphasis)
    print(y)
    mel = audio.melspectrogram(y, hp)
    print(mel)
    print(mel.shape)
    mfcc = audio.mfcc(y, hp)
    print(mfcc)
    print(mfcc.shape)
    '''
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
    '''
    S = librosa.feature.melspectrogram(y=y, sr=hp.sr, n_fft=hp.n_fft, 
                                   hop_length=hp.hop_length, 
                                   win_length=hp.win_length,
                                   n_mels=hp.n_mels,
                                   power=1) # S == mel when power == 1
    print(S)
    assert (mel == S).all()
    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel)) # 等价 mel = 2 * librosa.core.power_to_db(mel, amin=1e-5)
    mag = 20 * np.log10(np.maximum(1e-5, magnitude))
    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
	# Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    '''
    return mel, mag
    



if __name__ == "__main__":
    file = os.path.join(src_path, "S133_00324.wav")
    mel, mag = get_spectrograms(file)
    wav_data = spectrogram2wav(mag)
    wav_path = os.path.join(".", "rebuild.wav")
    sf.write(wav_path, wav_data, hp.sr, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
