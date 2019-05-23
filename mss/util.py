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

frame_length = 1024
frame_step = 512
fft_length = 1024

def to_channel_tensor(data, channel):
    return tf.transpose(tf.convert_to_tensor(data.astype(np.float32)))[channel]

def stft(audio):
    return tf.signal.stft(audio, 
                          frame_length=frame_length, 
                          frame_step=frame_step,
                          fft_length=fft_length)

def istft(data):
    return tf.signal.inverse_stft(
        stfts=data, 
        frame_length=frame_length,
        frame_step=frame_step,
        # forward_window_fn
        window_fn=tf.signal.inverse_stft_window_fn(
            frame_step=frame_step,
            forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)
        )
    )

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
