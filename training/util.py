"""Function to train a model."""
from time import time
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
# Hide lines below until Lab 4
import wandb
from wandb.keras import WandbCallback
# Hide lines above until Lab 4

from mss.datasets.dataset import Dataset
from mss.models.base import Model
from training.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = True
GPU_UTIL_SAMPLER = False


def train_model(
        model: Model,
        epochs: int,
        batch_size: int,
        gpu_ind: Optional[int] = None,
        use_wandb: bool = False) -> Model:
    """Train model."""
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilization = GPUUtilizationSampler(gpu_ind)
        callbacks.append(gpu_utilization)

    # Hide lines below until Lab 4
    if use_wandb:
        wandb.init()
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)
    # Hide lines above until Lab 4

    model.network.summary()

    t = time()
    _history = model.fit(batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print('Training took {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return model
