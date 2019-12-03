# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-12-03 16:52:10
@LastEditors: houwx
@LastEditTime: 2019-12-03 16:53:16
@Description: https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/utils.py
'''
import scipy.io.wavfile


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample
    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)