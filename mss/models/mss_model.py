"""Define CharacterModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from mss.models.base import Model
from mss.datasets import MUSDBDataset
from mss.networks.dnn import dnn
from mss.util import to_channel_tensor, stft, istft


class MSSModel(Model):
    def __init__(self,
                 dataset_cls: type = MUSDBDataset,
                 network_fn: Callable = dnn,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def separate_audio(self, audio):
        x = stft(to_channel_tensor(audio, 0))

        pred_raw = self.network.predict(x)
        separated_audio = istft(pred_raw)

        return separated_audio