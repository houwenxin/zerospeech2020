'''
Adapted From: https://github.com/Suhee05/Zerospeech2019/blob/master/utils/audio.py
'''

import scipy
import librosa
import numpy as np
from scipy.fftpack import dct

def load_wav(path, sr=16000):
    return librosa.core.load(path, sr=sr)[0] # Load an audio file as a floating point time series.

def preemphasis(x, preemphasis=0.97):
    #np.append(x[0], x[1:] - preemphasis * x[0:-1]) # 预加重：y(n)=x(n)-ax(n-1) 其中a为预加重系数，一般是0.9~1.0之间，通常取0.98。这里取0.97
    return scipy.signal.lfilter([1, -preemphasis], [1], x) # np.float64, More Accurate

def inv_preemphasis(x, preemphasis=0.97):
  return scipy.signal.lfilter([1], [1, -preemphasis], x)

def trim_silence(wav):
    # Trim leading and trailing silence from an audio signal.
	return librosa.effects.trim(wav)[0]

def mfcc(sample, hp):
    mfcc = _mfcc(sample, hp)
    mfcc_delta = _delta(mfcc)
    mfcc_delta_delta = _delta(mfcc_delta)
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta)).T
    return feature
def _delta(S):
    delta = librosa.feature.delta(S, width=7)
    return delta
def _mfcc(sample, hp):
    S = melspectrogram(sample, hp)
    dct_type=2
    norm='ortho'
    return dct(S, axis=0, type=dct_type, norm=norm)[:hp.n_mfcc] # TODO Verify: This should be equal to librosa.feature.mfcc(S, ...)
def melspectrogram(wav, hp):
	D = _stft(wav, hp)
	S = _amp_to_db(_linear_to_mel(np.abs(D) ** hp.magnitude_power, hp), hp) - hp.ref_level_db
	if hp.signal_norm:
		return _normalize(S, hp)
	return S

_mel_basis = None
def _linear_to_mel(spectogram, hp):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis(hp)
	return np.dot(_mel_basis, spectogram)
def _build_mel_basis(hp):
	assert hp.fmax <= hp.sr // 2
	return librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels,
							   fmin=hp.fmin, fmax=hp.fmax)
def _stft(y, hp):
    # Short-time Fourier transform (STFT). 
    # Return D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
    return librosa.stft(y=y, 
                        n_fft=hp.n_fft, 
                        hop_length=hp.hop_length, 
                        win_length=hp.win_length,
                        center=False)
def _amp_to_db(x, hp):
    #min_level = np.exp(hp.min_level_db / 20 * np.log(10)) # TODO How to calculate
    return 20 * np.log10(np.maximum(1e-5, x)) # Why 1e-5
def _normalize(S, hp):
    if hp.symmetric_mels:
	    return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / \
            (-hp.min_level_db)) - hp.max_abs_value, -hp.max_abs_value, hp.max_abs_value)
    else:
        return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
"""
############################################# UNUSED ########################################################################
def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hparams.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
        return x + hop_length
    return len(wav)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
    return y


def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
        est = _stft_tensorflow(y)
        angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
        y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)





def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


# Conversions:


def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)



def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
"""