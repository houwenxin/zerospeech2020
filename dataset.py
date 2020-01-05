# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-25 19:01:12
@LastEditors  : houwx
@LastEditTime : 2020-01-05 17:13:40
@Description: 
'''

import torch
import torch.nn.functional as F
import librosa
import numpy as np
import random
from pathlib import Path

def get_file_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    if "synthesis" in filename.name:
        files = [line.rstrip().split()[0] for line in lines]
    else:
        files = [line.rstrip() for line in lines]
    return files

def get_target_list(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    return [line.rstrip().split()[1] for line in lines]

class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_files, segment_length, sampling_rate, augment=True, mode='reconst'):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        #self.file_type = Path(audio_files).stem
        self.audio_files = get_file_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x if x.endswith(".wav") else Path(audio_files).parent / (x + ".wav") for x in self.audio_files]
        
        assert mode in ['reconst', 'convert'] # Reconstruction / Conversion
        self.mode = mode
        if self.mode == 'convert':
            self.target_speakers = get_target_list(audio_files)
            speakers = sorted(set(self.target_speakers))
            self.speaker2id = self.build_speaker2id(speakers)
            print("Target Speaker Dict: ", self.speaker2id)
            del speakers
        elif self.mode == 'reconst':
            speakers = sorted(set([Path(x).stem.split("_")[0] for x in self.audio_files]))
            #print(speakers)
            self.speaker2id = self.build_speaker2id(speakers)
            print("Speaker Dict: ", self.speaker2id)
            del speakers
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data
        
        speaker_info = {}
        source_speaker_name = Path(filename).stem.split("_")[0]
        #print(source_speaker_name)
        if source_speaker_name in self.speaker2id.keys():
            source_speaker = {"name": source_speaker_name, "id": self.speaker2id[source_speaker_name]}
        else:
            source_speaker = {"name": source_speaker_name, "id": -1}
        
        #print(source_speaker[0])
        
        speaker_info['source_speaker'] = source_speaker
        if self.mode == 'convert':
            target_speaker = {"name": self.target_speakers[index], "id": self.speaker2id[self.target_speakers[index]]}
            speaker_info['target_speaker'] = target_speaker
        #print(speaker_info)
        # audio = audio / 32768.0
        return audio.unsqueeze(0), speaker_info

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = librosa.load(full_path, sr=self.sampling_rate)
        data = 0.95 * librosa.util.normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate
    
    def build_speaker2id(self, speakers):
        speaker2id = {}
        for index, speaker in enumerate(speakers):
            speaker2id[speaker] = index
        return speaker2id
    
    def get_speaker_num(self):
        return len(self.speaker2id)