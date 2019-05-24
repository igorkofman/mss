"""CharacterPredictor class"""
from typing import Tuple, Union

import numpy as np

from .models import MSSModel
import mss.util as util
import argv
import soundfile as sf
import stempeg
import sys


class SourceSeparator:
    """Given an audio file, separates it"""
    def __init__(self):
        self.model = MSSModel()
        self.model.load_weights()
        self.samplerate = None

    def separate(self, audio_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(audio_or_filename, str):
            if (str.endswith(".stem.mp4")):
                audio, rate = stempeg.read_stems(
                    filename=audio_or_filename,
                    stem_id=5)
            else:
                audio, self.samplerate = sf.read(audio_or_filename)
        else:
            audio = audio_or_filename
        return self.model.separate_audio(audio)

    def evaluate(self, dataset):
        """Evaluate on a dataset."""
        return self.model.evaluate(dataset.x_test, dataset.y_test)

if __name__ == "__main__":
    separator = SourceSeparator()
    audio = separator.separate(sys.argv[1])
    sf.write(sys.argv[2], audio, separator.samplerate)
