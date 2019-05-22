#!/bin/bash
python training/run_experiment.py --save '{"dataset": "MUSDBDataset", "model": "MSSModel", "network": "dnn", "train_args": {"batch_size": 256}}'
