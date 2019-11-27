# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-18 15:15:56
@LastEditors: houwx
@LastEditTime: 2019-11-25 18:23:40
@Description: 
'''
import os
import numpy as np
import soundfile as sf
import h5py
import glob
from collections import defaultdict
import json

from hps.hps import hp
from utils import audio2 as audio
from utils.sampler import Sampler

src_path = "databases/english_small/train/unit"
trg_path = "./"

def get_spectrograms(sound_file):
	y = audio.load_wav(sound_file, sr=hp.sr)
	#print(len(y)
	y = audio.trim_silence(y)
	#print(y)
	y = audio.preemphasis(y, hp.preemphasis)
	#print(y)
	hp.rescale = False
	if hp.rescale: # y is too small here.
		y = audio.rescale(y, hp)
		#Assert all audio is in [-1, 1]
		if (y > 1.).any() or (y < -1.).any():
			raise RuntimeError('wav has invalid value: {}'.format(sound_file))
	#print(y)
	mag = audio.linear_spectrogram(y, hp)
	#print(mag)
	#print(mag.shape)
	mel = audio.linear2mel(mag, hp)
	#print(mel)
	#print(mel.shape)
	mel = audio.amp2db(mel) # 等价 mel = 2 * librosa.core.power_to_db(mel, amin=1e-5)
	mag = audio.amp2db(mag)
	mel = audio.normalize(mel, hp)
	mag = audio.normalize(mag, hp)
	
	# Compute MFCCs
	mfcc = audio.mel2mfcc(mel, hp)

	mel = mel.T.astype(np.float32)  # (T, n_mels)
	mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
	mfcc = mfcc.T.astype(np.float32) # (T, 3 * n_mfcc)

	return mag, mel, mfcc

def preprocess(source_path,
			   target_path,
			   test_path,
			   dataset_path, 
			   index_path, 
			   index_source_path, 
			   index_target_path, 
			   speaker2id_path,
			   seg_len=128, 
			   n_samples=200000,
			   dset='train',
			   remake=False):
	
	if remake or not os.path.isfile(dataset_path):
		with h5py.File(dataset_path, 'w') as h5py_file:
			grps = [h5py_file.create_group('train'), h5py_file.create_group('test')]
			print('[Processor] - making training dataset...')
			make_dataset(grps, seg_len, root_dir=source_path)
			make_dataset(grps, seg_len, root_dir=target_path)
			print('[Processor] - making testing dataset...')
			make_dataset(grps, seg_len, root_dir=test_path, make_test=True, pad=False)

	# stage 1 training samples
	print('[Processor] - making stage 1 training samples with segment length = ', seg_len)
	make_samples(dataset_path, index_path, speaker2id_path,
				make_object='all',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

	# stage 2 training source samples
	print('[Processor] - making stage 2 training source samples with segment length = ', seg_len)
	make_samples(dataset_path, index_source_path, speaker2id_path,
				make_object='source',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

	# stage 2 training target samples
	print('[Processor] - making stage 2 training target samples with segment length = ', seg_len)
	make_samples(dataset_path, index_target_path, speaker2id_path,
				make_object='target',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

def make_dataset(grps, seg_len, root_dir, make_test=False, pad=True):
    	
	filenames = glob.glob(os.path.join(root_dir, '*_*.wav')) 
	filenames = [filename.replace("\\", "/") for filename in filenames] # In case run in Windows, replace "\\" with "/" to prevent error.

	filename_groups = defaultdict(lambda : []) # Format: {"V001":[path/xx_x.wav, path/yy_y.wav], "V002":[path/x_xxx.wav, path/y_yyy.wav]}

	for filename in filenames:
		# divide into groups
		speaker_id, segment_id = filename.strip().split('/')[-1].strip('.wav').split('_')
		filename_groups[speaker_id].append(filename)
	
	print('Number of speakers: ', len(filename_groups))
	grp = grps[1] if make_test else grps[0]

	for speaker_id, filenames in filename_groups.items():
		for filename in filenames: # Single audio file.
			speaker_id, segment_id = filename.strip().split('/')[-1].strip('.wav').split('_')
			lin_spec, mel_spec, mfcc_spec = get_spectrograms(filename)

			if pad and len(lin_spec) <= seg_len:
				mel_padding = np.zeros((seg_len - mel_spec.shape[0] + 1, mel_spec.shape[1]))
				lin_padding = np.zeros((seg_len - lin_spec.shape[0] + 1, lin_spec.shape[1]))
				mfcc_padding = np.zeros((seg_len - mfcc_spec.shape[0] + 1, mfcc_spec.shape[1]))
				mel_spec = np.concatenate((mel_spec, mel_padding), axis=0)
				lin_spec = np.concatenate((lin_spec, lin_padding), axis=0)
				mfcc_spec = np.concatenate((mfcc_spec, mfcc_padding), axis=0)
				print('[Processor] - processing {}: {} - padded to {}'.format(speaker_id, filename, np.shape(lin_spec)), end='\r')
			else:
				print('[Processor] - processing {}: {}'.format(speaker_id, filename), end='\r')
				
			grp.create_dataset('{}/{}/mel'.format(speaker_id, segment_id), data=mel_spec, dtype=np.float32)
			grp.create_dataset('{}/{}/lin'.format(speaker_id, segment_id), data=lin_spec, dtype=np.float32)
			grp.create_dataset('{}/{}/mfcc'.format(speaker_id, segment_id), data=mfcc_spec, dtype=np.float32)
		print()
	print()


def make_samples(h5py_path, 
				 json_path, 
				 speaker2id_path, 
				 make_object, 
				 seg_len=64, 
				 n_samples=200000, 
				 dset='train'):

	sampler = Sampler(h5py_path, dset, seg_len, speaker2id_path, make_object)
	samples = [sampler.sample()._asdict() for _ in range(n_samples)]
	with open(json_path, 'w') as f_json:
		json.dump(samples, f_json, indent=4, separators=(',', ': '))
        
		
if __name__ == "__main__":
	file = os.path.join(src_path, "S133_00324.wav")
	mag, mel, mfcc = get_spectrograms(file)
	print(mag.shape)
	print(mel.shape)
	print(mfcc.shape)
	#wav_data_lin = audio.linear2wav(mag, hp)
	#wav_data_mel = audio.mel2wav(mel, hp)
	#wav_path1 = os.path.join(".", "rebuild_lin.wav")
	#wav_path2 = os.path.join(".", "rebuild_mel.wav")
	#sf.write(wav_path1, wav_data_lin, hp.sr, 'PCM_16') # import soundfile as sf, conda下安装不了，得用pip装
	#sf.write(wav_path2, wav_data_mel, hp.sr, 'PCM_16')
	'''
	preprocess(source_path="./databases/english_small/train/unit/",
			   target_path="./databases/english_small/train/voice/",
			   test_path="./databases/english_small/test/",
			   dataset_path="./dataset/english/dataset.hdf5", 
			   index_path="./dataset/english/index.json", 
			   index_source_path="./dataset/english/index_src.json", 
			   index_target_path="./dataset/english/index_trg.json", 
			   speaker2id_path="./dataset/english/speaker2id.json",
			   seg_len=128, 
			   n_samples=200,
			   dset='train',
			   remake=False)
	'''