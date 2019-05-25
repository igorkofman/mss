"""
MUSDB dataset. Downloads from MUSDB website and saves as .npz file if not already present.
"""
import os
import zipfile
import musdb
import numpy as np
import tensorflow as tf
from mss.datasets.dataset import _download_raw_dataset, Dataset, _parse_args
from mss.util import to_channel_tensor, stft

#from boltons.cacheutils import cachedproperty
#from tensorflow.keras.utils import to_categorical
#import h5py
#import numpy as np
#import toml

class MUSDBDataset(Dataset):
    """
    Blah!
    """
    def __init__(self, subsample_fraction: float = None):
        self._ensure_dataset_exists_locally()
        self.database = musdb.DB(self.data_dirname() / 'musdb18')
        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.input_shape = (513*2,)
        self.output_shape = (513*2,)

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

        train_raw_x = to_channel_tensor(train_tracks[0].audio, 0)
        train_raw_y = to_channel_tensor(train_tracks[0].stems[0], 0)
        test_raw_x = to_channel_tensor(test_tracks[0].audio, 0)
        test_raw_y = to_channel_tensor(test_tracks[0].stems[0], 0)

        self.x_train = stft(train_raw_x)
        self.y_train = stft(train_raw_y)
        self.x_test = stft(test_raw_x)
        self.y_test = stft(test_raw_y)

        self._subsample()

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        self.x_train = self.x_train[:num_train]
        self.y_train = self.y_train[:num_train]
        self.x_test = self.x_test[:num_test]
        self.y_test = self.y_test[:num_test]

    def __repr__(self):
        return ('MUSDB Dataset\n')

def main():
    """Load MUSDB dataset and print info."""
    args = _parse_args()
    dataset = MUSDBDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()

    print(dataset)
    print(dataset.x_train.shape, dataset.y_train.shape)  # pylint: disable=E1101
    print(dataset.x_test.shape, dataset.y_test.shape)  # pylint: disable=E1101


if __name__ == '__main__':
    main()
