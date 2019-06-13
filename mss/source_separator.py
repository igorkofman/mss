"""CharacterPredictor class"""
from typing import Tuple, Union
import argparse

import numpy as np

from mss.models import MSSModel
import mss.util as util
import sys 
import soundfile as sf
import stempeg
import sys
import tensorflow as tf

class SourceSeparator:
    """Given an audio file, separates it"""
    def __init__(self, num_leading_ctx_frames, num_trailing_ctx_frames):
        dataset_args = dict(num_leading_ctx_frames=num_leading_ctx_frames, num_trailing_ctx_frames=num_trailing_ctx_frames, batch_size=256)
        self.model = MSSModel(dataset_args=dataset_args)
        self.model.load_weights()
        self.samplerate = None
        self.test_mode = False


    def separate(self, audio_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(audio_or_filename, str):
            if audio_or_filename.endswith(".stem.mp4"):
                audio, self.samplerate = stempeg.read_stems(
                    filename=audio_or_filename,
                    stem_id=0)
            else:
                audio, self.samplerate = sf.read(audio_or_filename)
        else:
            audio = audio_or_filename
        if self.test_mode:
            return self.model.separate_audio_noop(audio)
        else:
            return self.model.separate_audio(audio)

    #stem_ids:
    #    mixture: 0
    #    drums: 1
    #    bass: 2
    #    other: 3
    #    vocals: 4

    def evaluate(self, dataset):
        """Evaluate on a dataset."""
        return self.model.evaluate(dataset.x_test, dataset.y_test)

def main():
    separator = SourceSeparator(1, 1)
    separator.test_mode = sys.argv[3] and sys.argv[3] == '-t'
    audio = separator.separate(sys.argv[1])
    sf.write(sys.argv[2], audio, separator.samplerate)

if __name__ == "__main__":
    main()
