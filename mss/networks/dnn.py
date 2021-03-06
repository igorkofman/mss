from typing import Tuple
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense


def dnn(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int = 2048,
        dropout_amount: float = 0.2,
        num_layers: int = 4) -> Model:

    model = Sequential()
    model.add(Dense(layer_size, input_shape=input_shape, activation='relu'))
    for _ in range(num_layers-1):
        model.add(Dense(layer_size, activation='relu'))
        # model.add(Dropout(dropout_amount))
    model.add(Dense(output_shape[0], activation=None))
    return model
