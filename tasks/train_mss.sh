#!/bin/bash
python training/run_experiment.py --save '{"dataset": "MUSDBDataset", "model": "MSSModel", "network": "dnn", "model_args":{"num_leading_frames": 10, "num_trailing_frames":10}, "train_args": {"batch_size": 256}}'
