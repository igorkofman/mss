"""CharacterPredictor class"""
from typing import Tuple, Union

import numpy as np

from .models import MSSModel
import text_recognizer.util as util


class SourceSeparator:
    """Given an audio file, separates it"""
    def __init__(self):
        self.model = MSSModel()
        self.model.load_weights()

    def separate(self, audio_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(audio_or_filename, str):
            audio = util.read_image(audio_or_filename, grayscale=True)
        else:
            audio = audio_or_filename
        return self.model.separate_audio(audio)

    def evaluate(self, dataset):
        """Evaluate on a dataset."""
        return self.model.evaluate(dataset.x_test, dataset.y_test)
