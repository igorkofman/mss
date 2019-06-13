"""
MUSDB dataset. Downloads from MUSDB website and saves as .npz file if not already present.
"""
import os
import random
import math
import zipfile
import musdb
from tensorflow.python.keras.utils.data_utils import Sequence

from mss.datasets.dataset import _download_raw_dataset, Dataset, _parse_args
from mss.util import to_channel_tensor, stft, pad, built_contextual_frames

class MUSDBDataset(Dataset, Sequence):
    """
    Tunes!
    """
    def __init__(self, batch_size=32, num_leading_frames=1, num_trailing_frames=1,
                 frame_length=1024, frame_step=512, target_stem_id=1):

        self._ensure_dataset_exists_locally()
        self.database = musdb.DB(self.data_dirname() / 'musdb18')

        # configuration
        self.batch_size = batch_size
        self.num_leading_frames = num_leading_frames
        self.num_trailing_frames = num_trailing_frames
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.target_stem_id = target_stem_id

        # expose dataset args for network shape
        num_fft_bins = (self.frame_length // 2) + 1
        frame_len_with_context = self.num_leading_frames + self.num_trailing_frames + 1
        self.input_shape = (num_fft_bins * 2 * frame_len_with_context,)
        self.output_shape = (num_fft_bins * 2,)

        # internal state
        self.train_tracks = None
        self.test_tracks = None
        self.track_idx = None
        self.current_track = None
        self.padded_x_stft = None
        self.padded_y_stft = None

    def __len__(self):
        return sum([
            math.ceil((t.duration * t.rate // self.frame_step) / self.batch_size) 
            for t in self.train_tracks])

    def on_epoch_end(self):
        random.shuffle(self.train_tracks)
        self._set_current_track(0)

    def _set_current_track(self, track_idx):
        self.track_idx = track_idx
        self.current_track = self.train_tracks[self.track_idx]
        print (self.current_track)
        x_audio = to_channel_tensor(self.current_track.audio, 0)
        y_audio = to_channel_tensor(self.current_track.stems[self.target_stem_id], 0)
        print (x_audio)
        x_stft = stft(x_audio, self.frame_length, self.frame_step)
        print (x_stft)
        y_stft = stft(y_audio, self.frame_length, self.frame_step)
        self.padded_x_stft = pad(x_stft, self.num_leading_frames, self.num_trailing_frames)
        print (self.padded_x_stft)
        self.padded_y_stft = pad(y_stft, self.num_leading_frames, self.num_trailing_frames)

    def __getitem__(self, iter_index):
        # todo: vectorize

        # if we're out of data in the current track, advance to the next track
        samples_in_current_track = self.current_track.duration * self.current_track.rate // self.frame_step
        if iter_index * self.batch_size >= samples_in_current_track:
            self._set_current_track(self.track_idx+1)

        # batch_start and batch_end, adjusted for the padding and clipped at end of data
        batch_start = iter_index * self.batch_size + self.num_leading_frames
        batch_end = int(min((iter_index + 1) * self.batch_size, samples_in_current_track) + \
                        self.num_leading_frames)

        x_with_context = built_contextual_frames(self.padded_x_stft,
                                                 batch_start, batch_end,
                                                 self.num_leading_frames,
                                                 self.num_trailing_frames)

        y_batch = self.padded_y_stft[batch_start:batch_end, :]
        return (x_with_context, y_batch)

    def _ensure_dataset_exists_locally(self):
        musdb_datadir = self.data_dirname() / 'musdb18'
        musdb_datafile = self.data_dirname() / 'musdb18.zip'
        if not os.path.exists(musdb_datadir):
            if not os.path.exists(musdb_datafile):
                _download_raw_dataset({
                    'url': 'https://s3-us-west-2.amazonaws.com/igor-ml/musdb18.zip',
                    'filename': musdb_datafile,
                    'sha256': '2765dd51fed264bbe2dac5c382f7120b7cf1932698fc67d100aa0f3811479152',
                })
            curdir = os.getcwd()
            os.chdir(MUSDBDataset.data_dirname())
            zip_file = zipfile.ZipFile(musdb_datafile, 'r')
            zip_file.extractall('musdb18')
            os.chdir(curdir)

    def load_or_generate_data(self):
        self.train_tracks = self.database.load_mus_tracks(subsets=['train'])
        self.test_tracks = self.database.load_mus_tracks(subsets=['test'])
        self._set_current_track(0)

        # grab just the first track for now
        # later we'll want to computed the length of all tracks and do ffts on the fly
        

    def __repr__(self):
        return ('MUSDB Dataset\n')

def main():
    """Load MUSDB dataset and print info."""
    args = _parse_args()
    dataset = MUSDBDataset()
    dataset.load_or_generate_data()

    print(dataset)
    print(dataset.x_train.shape, dataset.y_train.shape)  # pylint: disable=E1101
    print(dataset.x_test.shape, dataset.y_test.shape)  # pylint: disable=E1101


if __name__ == '__main__':
    main()
