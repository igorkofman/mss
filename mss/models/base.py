"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional

from tensorflow.keras.optimizers import Adam

DIRNAME = Path(__file__).parents[1].resolve() / 'weights'

class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, network_args: Dict = None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

    def fit(self, dataset, steps_per_epoch: int = 100, epochs: int = 16, augment_val: bool = True,
            callbacks: list = None):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        self.network.fit(dataset.x_train, dataset.x_train, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    def evaluate(self, x, y, steps=100, verbose=False):  # pylint: disable=unused-argument
        return self.network.evaluate(x, x, steps=10)
        
    def loss(self):  # pylint: disable=no-self-use
        return 'mean_squared_error' #'kullback_leibler_divergence'

    def optimizer(self):  # pylint: disable=no-self-use
        return Adam()

    def metrics(self):  # pylint: disable=no-self-use
        return []

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
