# -*- coding: UTF-8 -*-
'''
@Author: houwx
@Date: 2019-11-22 15:48:19
@LastEditors: houwx
@LastEditTime: 2019-11-22 16:45:02
@Description: 
'''
import h5py
import random
import numpy as np
from collections import namedtuple
from collections import defaultdict
import json

class Sampler(object):
	def __init__(self, 
				h5py_path,
				dset='train', 
				seg_len=64,
				speaker2id_path='',
				make_object='all'):
		self.dset = dset
		self.f_h5 = h5py.File(h5py_path, 'r')
		self.seg_len = seg_len
		self.speaker2id_path = speaker2id_path
		if 'english' in h5py_path:
			self.target_speakers = ['V001', 'V002']
		elif 'surprise' in h5py_path:
			self.target_speakers = ['V001']
		else:
			raise NotImplementedError('Invalid dataset.hdf5 name!')

		if make_object == 'all': 
			self.speaker_used = sorted(list(self.f_h5[dset].keys()))
			self.save_speaker2id()
			print('[Sampler] - Generating stage 1 training segments...')
		elif make_object == 'source':
			self.get_speaker2id()
			self.speaker_used = [s for s in sorted(list(self.f_h5[dset].keys())) if s not in self.target_speakers]
			print('[Sampler] - Generating stage 2 training source segments...')
		elif make_object == 'target':
			self.get_speaker2id()
			self.speaker_used = self.target_speakers
			print('[Sampler] - Generating stage 2 training target segments...')
		else:
			raise NotImplementedError('Invalid make object!')
		print('[Sampler] - Speaker used: ', self.speaker_used)

		self.speaker2utts = {speaker : sorted(list(self.f_h5[f'{dset}/{speaker}'].keys())) for speaker in self.speaker_used}
		self.rm_too_short_utt()
		self.speaker_weight = [len(self.speaker2utts[speaker_id]) / self.total_utt for speaker_id in self.speaker_used]
		self.indexer = namedtuple('index', ['speaker', 'i', 't'])

	
	def get_num_utts(self):
		cnt = 0
		for speaker_id in self.speaker_used: cnt += len(self.speaker2utts[speaker_id])
		return cnt
		

	def rm_too_short_utt(self, limit=None):
		self.total_utt = self.get_num_utts()
		to_rm = defaultdict(lambda : [])
		if limit is None:
			limit = self.seg_len
		for speaker_id in self.speaker_used:
			for utt_id in self.speaker2utts[speaker_id]:
				if self.f_h5[f'{self.dset}/{speaker_id}/{utt_id}/lin'].shape[0] <= limit:
					to_rm[speaker_id].append(utt_id)
		for speaker_id, utt_ids in to_rm.items():
			for utt_id in utt_ids:
				self.speaker2utts[speaker_id].remove(utt_id)
		new_cnt = self.get_num_utts()
		print('[Sampler] - %i too short utterences out of a total of %i are removed.' % (self.total_utt - new_cnt, self.total_utt))

				
	def sample_utt(self, speaker_id, n_samples=1):
		# sample an utterence
		utt_ids = random.sample(self.speaker2utts[speaker_id], n_samples)
		lengths = [self.f_h5[f'{self.dset}/{speaker_id}/{utt_id}/lin'].shape[0] for utt_id in utt_ids]
		return [(utt_id, length) for utt_id, length in zip(utt_ids, lengths)]


	def rand(self, l):
		rand_idx = random.randint(0, len(l) - 1)
		return l[rand_idx] 


	def sample(self):
		speaker = np.random.choice(self.speaker_used, p=self.speaker_weight)
		speaker_idx = self.speaker2id[speaker]
		(utt_id, utt_len), = self.sample_utt(speaker, 1)
		t = random.randint(0, utt_len - self.seg_len)  
		index_tuple = self.indexer(speaker=speaker_idx, i=f'{speaker}/{utt_id}', t=t)
		return index_tuple
		

	def save_speaker2id(self):
		self.speaker2id = {speaker:i for i, speaker in enumerate(self.speaker_used)}
		with open(self.speaker2id_path, 'w') as file:
			file.write(json.dumps(self.speaker2id))


	def get_speaker2id(self):
		with open(self.speaker2id_path, 'r') as f_json:
			self.speaker2id = json.load(f_json)
