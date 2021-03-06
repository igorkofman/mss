"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional

from tensorflow.keras.optimizers import Adam

DIRNAME = Path(__file__).parents[1].resolve() / 'weights'

class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, network_args: Dict = None, train_args: Dict = None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.dataset = dataset_cls(**dataset_args)
        self.dataset.load_or_generate_data()

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.dataset.input_shape, self.dataset.output_shape, **network_args)
        self.network.summary()

        if train_args is not None:
            self.learning_rate = train_args['learning_rate']

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

    def fit(self, batch_size: int = 100, epochs: int = 16, callbacks: list = None):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        self.dataset.batch_size = batch_size
        self.network.fit_generator(self.dataset, epochs=epochs, shuffle=False, verbose=1)

    def evaluate(self):  # pylint: disable=unused-argument
        pass
#        return self.network.evaluate_generator(dataset, steps=10)
        
    def loss(self):  # pylint: disable=no-self-use
        return 'mean_squared_error'

    def optimizer(self):  # pylint: disable=no-self-use
        if self.learning_rate:
            return Adam(lr=self.learning_rate)
        else:
            return Adam()

    def metrics(self):  # pylint: disable=no-self-use
        return []

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
