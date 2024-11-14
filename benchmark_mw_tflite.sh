#!/usr/bin/env bash

source .venv/bin/activate

DEFAULT_N_IMAGES=20

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 ./run_mw_tflite.py --threads 1 --dataset_size $N_IMAGES
python3 ./run_mw_tflite.py --threads 4 --dataset_size $N_IMAGES

deactivate
