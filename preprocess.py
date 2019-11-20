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

import soundfile as sf
from utils import audio2 as audio

src_path = "../../databases/english_small/train/unit"
trg_path = "./"

def get_spectrograms(sound_file):
    y = audio.load_wav(sound_file, sr=hp.sr)
    #print(len(y)
    y = audio.trim_silence(y)
    print(y)
    y = audio.preemphasis(y, hp.preemphasis)
    #print(y)
    hp.rescale = False
    if hp.rescale: # y is too small here.
        y = audio.rescale(y, hp)
        #Assert all audio is in [-1, 1]
        if (y > 1.).any() or (y < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(sound_file))
    print(y)
    mag = audio.linear_spectrogram(y, hp)
    print(mag)
    print(mag.shape)
    mel = audio.linear2mel(mag, hp)
    #print(mel)
    #print(mel.shape)
    mfcc = audio.mel2mfcc(y, hp)

    mel = audio.amp2db(mel) # 等价 mel = 2 * librosa.core.power_to_db(mel, amin=1e-5)
    mag = audio.amp2db(mag)
    mel = audio.normalize(mel, hp)
    mag = audio.normalize(mag, hp)
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    return mag, mel, mfcc

if __name__ == "__main__":
    file = os.path.join(src_path, "S133_00324.wav")
    mag, mel, mfcc = get_spectrograms(file)
    wav_data_lin = audio.linear2wav(mag, hp)
    wav_data_mel = audio.mel2wav(mel, hp)
    wav_path1 = os.path.join(".", "rebuild_lin.wav")
    wav_path2 = os.path.join(".", "rebuild_mel.wav")
    sf.write(wav_path1, wav_data_lin, hp.sr, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
    sf.write(wav_path2, wav_data_mel, hp.sr, 'PCM_16')
