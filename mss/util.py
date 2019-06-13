"""Utility functions for text_recognizer module."""
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
from urllib.request import urlopen, urlretrieve
import hashlib
import functools
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# pads a tensor of fft'd audio samples with the appopriate number of
# leading and trailing frames of silence
def pad(tensor, num_leading_ctx_frames, num_trailing_ctx_frames):
    paddings = tf.constant([[num_leading_ctx_frames, num_trailing_ctx_frames], [0, 0]])
    return tf.pad(tensor, paddings, 'CONSTANT')

# given a tensor of padded fft'd audio data returns a tensor of frames with leading and trailing context
# padded_data must be valid at [first_frame-num_leading_ctx_frames]  and at [first_frame+num_frames+num_trailing_ctx_frames-1]
def built_contextual_frames(padded_data, batch_start, batch_end, num_leading_ctx_frames, num_trailing_ctx_frames):
    out = []
    for i in range(batch_start, batch_end):
        row = padded_data[i - num_leading_ctx_frames : i + num_trailing_ctx_frames + 1, :]
        new_frame_width = num_leading_ctx_frames + num_trailing_ctx_frames + 1
        row = tf.reshape(row, [1, row.shape[1] * new_frame_width])
        out.append(row)
    return tf.concat(out, axis=0)

def to_channel_tensor(data, channel):
    # convert to float32 because that's what the stft wants
    # data is (timestamp, channel)
    return tf.convert_to_tensor(data.astype(np.float32))[:, channel]

def stft(audio, frame_length, frame_step):
    x = tf.abs(tf.signal.stft(audio,
                       frame_length=frame_length,
                       frame_step=frame_step))
    x_real = tf.math.real(x)
    x_imag = tf.math.imag(x)
    res = tf.concat([x_real, x_imag], axis=1)
    return res

def istft(data, frame_length, frame_step):
    sess = tf.Session()
    data = tf.convert_to_tensor(data)
    shape = data.get_shape().as_list()
    data = tf.complex(data[:, :int(shape[1] / 2)], data[:, int (shape[1] / 2):])
    res = tf.signal.inverse_stft(
        stfts=data, 
        frame_length=frame_length,
        frame_step=frame_step,
        # forward_window_fn
        window_fn=tf.signal.inverse_stft_window_fn(
            frame_step=frame_step,
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
