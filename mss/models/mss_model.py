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
                 train_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args, train_args)
        self.num_leading_ctx_frames = dataset_args['num_leading_ctx_frames']
        self.num_trailing_ctx_frames = dataset_args['num_trailing_ctx_frames']
        
    def separate_audio(self, audio):
        frames = pad(stft(to_channel_tensor(audio, 0)), self.num_leading_ctx_frames, self.num_trailing_ctx_frames)
        padded_length = frames.shape[0]
        batch_start = self.num_leading_ctx_frames
        batch_end = padded_length-self.num_leading_ctx_frames-self.num_trailing_ctx_frames
        frames_with_context = built_contextual_frames(frames, batch_start, batch_end,
            self.num_leading_ctx_frames, self.num_trailing_ctx_frames)
        pred_raw = self.network.predict(frames_with_context, steps=1)
        separated_audio = istft(pred_raw)
        return separated_audio
    
    def separate_audio_noop(self, audio):
        pred_raw = stft(to_channel_tensor(audio, 0))
        separated_audio = istft(pred_raw)
        return separated_audio
