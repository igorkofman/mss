"""
MUSDB dataset. Downloads from MUSDB website and saves as .npz file if not already present.
"""
import os
import zipfile
import musdb
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence

from mss.datasets.dataset import _download_raw_dataset, Dataset, _parse_args
from mss.util import to_channel_tensor, stft, pad, built_contextual_frames

class MUSDBDataset(Dataset, Sequence):
    """
    Tunes!
    """
    def __init__(self, batch_size, num_fft_bins=513, num_leading_ctx_frames=1, num_trailing_ctx_frames=1):
        self._ensure_dataset_exists_locally()
        self.database = musdb.DB(self.data_dirname() / 'musdb18')
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.batch_size = batch_size
        self.num_samples = 0
        self.num_fft_bins = num_fft_bins
        self.num_leading_ctx_frames = num_leading_ctx_frames
        self.num_trailing_ctx_frames = num_trailing_ctx_frames
        self.frame_len_with_context = num_leading_ctx_frames + num_trailing_ctx_frames + 1
        self.input_shape = (self.num_fft_bins * 2 * self.frame_len_with_context,)
        self.output_shape = (self.num_fft_bins * 2,)

    def __len__(self):
        return int(self.num_samples / self.batch_size)

    def __getitem__(self, iter_index):
        # todo: vectorize
        batch_start = iter_index * self.batch_size + self.num_leading_ctx_frames
        batch_end = min((iter_index + 1) * self.batch_size, 
                self.num_samples - self.num_trailing_ctx_frames) + \
                self.num_leading_ctx_frames
        x_with_context = built_contextual_frames(pad(self.x_train, self.num_leading_ctx_frames, self.num_trailing_ctx_frames), 
                                                 batch_start, batch_end,
                                                 self.num_leading_ctx_frames,
                                                 self.num_trailing_ctx_frames)
        y_batch = self.y_train[batch_start:batch_end]

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
        train_tracks = self.database.load_mus_tracks(subsets=['train'])
        test_tracks = self.database.load_mus_tracks(subsets=['test'])

        # grab just the first track for now
        # later we'll want to computer the length of all tracks and do ffts on the fly
        train_raw_x = to_channel_tensor(train_tracks[0].audio, 0)
        train_raw_y = to_channel_tensor(train_tracks[0].stems[0], 0)
        test_raw_x = to_channel_tensor(test_tracks[0].audio, 0)
        test_raw_y = to_channel_tensor(test_tracks[0].stems[0], 0)

        self.x_train = stft(train_raw_x)
        self.y_train = stft(train_raw_y)
        self.x_test = stft(test_raw_x)
        self.y_test = stft(test_raw_y)
        self.num_samples = int(self.x_train.shape[0])

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
