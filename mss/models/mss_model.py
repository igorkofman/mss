"""Define CharacterModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from mss.models.base import Model
from mss.datasets import MUSDBDataset
from mss.networks.dnn import dnn


class MSSModel(Model):
    def __init__(self,
                 dataset_cls: type = MUSDBDataset,
                 network_fn: Callable = dnn,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        # NOTE: integer to character mapping dictionary is self.data.mapping[integer]
        # Your code below (Lab 1)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_character = self.data.mapping[ind]
        # Your code above (Lab 1)
        return predicted_character, confidence_of_prediction
