"""
MUSDB dataset. Downloads from MUSDB website and saves as .npz file if not already present.
"""
import os
import zipfile
import musdb
from mss.datasets.dataset import _download_raw_dataset, Dataset, _parse_args
from mss.util import to_channel_tensor, stft
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence


NUM_FFT_BINS = 513

class MUSDBDataset(Dataset, Sequence):
    """
    Tunes!
    """
    def __init__(self, batch_size: int=32, subsample_fraction: float = None):
        self._ensure_dataset_exists_locally()
        self.database = musdb.DB(self.data_dirname() / 'musdb18')
        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.batch_size = batch_size
        self.num_samples = 0
        self.input_shape = (NUM_FFT_BINS*2*3,)
        self.output_shape = (NUM_FFT_BINS*2,)

    def __len__(self):
        return int(self.num_samples / self.batch_size)

    def __getitem__(self, iter_index):
        print("Iter index is:" + str(iter_index))
        # todo: vectorize
        batch_start = iter_index * self.batch_size
        batch_end = min((iter_index + 1) * self.batch_size, self.num_samples)
        x_batch = self.x_train[batch_start:batch_end]
        y_batch = self.y_train[batch_start+1:batch_end-1]

        out = []
        for i in range(1, self.batch_size-1):
            item = tf.concat([x_batch[i-1:i, :], x_batch[i:i+1, :], x_batch[i+1:i+2, :]], axis=1)
            out.append(item)
        x_with_context = tf.concat(out, axis=0)
        print(x_with_context)
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
        train_raw_x = to_channel_tensor(train_tracks[0].audio, 0)
        train_raw_y = to_channel_tensor(train_tracks[0].stems[0], 0)
        test_raw_x = to_channel_tensor(test_tracks[0].audio, 0)
        test_raw_y = to_channel_tensor(test_tracks[0].stems[0], 0)

        self.x_train = stft(train_raw_x)
        self.y_train = stft(train_raw_y)
        self.x_test = stft(test_raw_x)
        self.y_test = stft(test_raw_y)
        self.num_samples = int(self.x_train.shape[0]) 
        print (self.num_samples)

        #self._subsample()

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
