"""Define CharacterModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from mss.models.base import Model
from mss.datasets import MUSDBDataset
from mss.networks.dnn import dnn
from mss.util import to_channel_tensor, stft, istft, pad, built_contextual_frames


class MSSModel(Model):
    def __init__(self,
                 dataset_cls: type = MUSDBDataset,
                 network_fn: Callable = dnn,
                 dataset_args: Dict = None,
                 network_args: Dict = None,
                 train_args: Dict = None,
                 model_args: Dict = None):
        """Define the default dataset and network values for this model."""

        self.num_leading_frames = model_args['num_leading_frames']
        self.num_trailing_frames  = model_args['num_trailing_frames']
        self.frame_length = model_args['frame_length']
        self.frame_step = model_args['frame_step']

        # propagate model args to dataset
        dataset_args['num_leading_frames'] = self.num_leading_frames
        dataset_args['num_trailing_frames'] = self.num_trailing_frames
        dataset_args['frame_length'] = self.frame_length
        dataset_args['frame_step'] = self.frame_step

        super().__init__(dataset_cls, network_fn, dataset_args, network_args, train_args)

    def separate_audio(self, audio):
        x_stft = stft(to_channel_tensor(audio, 0), self.frame_length, self.frame_step)
        padded_x_stft = pad(x_stft, self.num_leading_frames, self.num_trailing_frames)
        batch_start = self.num_leading_frames
        batch_end = padded_x_stft.shape[0] - self.num_trailing_frames
        frames_with_context = built_contextual_frames(padded_x_stft, batch_start, batch_end,
                                                      self.num_leading_frames, self.num_trailing_frames)
        pred_raw = self.network.predict(frames_with_context, steps=1)
        separated_audio = istft(pred_raw, self.frame_length, self.frame_step)
        return separated_audio

    def separate_audio_noop(self, audio):
        pred_raw = stft(to_channel_tensor(audio, 0), self.frame_length, self.frame_step)
        separated_audio = istft(pred_raw, self.frame_length, self.frame_step)
        return separated_audio
