"""Utility functions for text_recognizer module."""
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
from urllib.request import urlopen, urlretrieve
import hashlib
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import functools

FRAME_LENGTH = 1024
FRAME_STEP = 512
FFT_LENGTH = 1024
NUM_FFT_BINS = int(FRAME_LENGTH/2) + 1
def to_channel_tensor(data, channel):
    return tf.transpose(tf.convert_to_tensor(data.astype(np.float32)))[channel]

def stft(audio):
    x = tf.signal.stft(audio,
                       frame_length=FRAME_LENGTH, 
                       frame_step=FRAME_STEP,
                       fft_length=FFT_LENGTH)
    x_real = tf.math.real(x)
    x_imag = tf.math.imag(x)
#    stacked = tf.stack([x_real, x_imag])
   
#    shape = stacked.get_shape().as_list()
#    res = tf.reshape(stacked, [shape[0], shape[1]*shape[2]])
    res = tf.concat([x_real, x_imag], axis=1)
    return res

def istft(data):
    sess = tf.Session()
    data = tf.convert_to_tensor(data)
    shape = data.get_shape().as_list()
#    data = tf.reshape(data, [shape[0], int(shape[1]/2), 2]) # check me
    data = tf.complex(data[:,0:513], data[:,:,513:])
    res = tf.signal.inverse_stft(
        stfts=data, 
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        # forward_window_fn
        window_fn=tf.signal.inverse_stft_window_fn(
            frame_step=FRAME_STEP,
            forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)
        )
    )
    return sess.run(res)

def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""
    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        blocks : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


# Hide lines below until Lab 7
def download_urls(urls, filenames):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(urlretrieve, url, filename) for url, filename in zip(urls, filenames)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print('Error', e)
# Hide lines above until Lab 7
