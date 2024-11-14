#!/usr/bin/env bash

source .venv/bin/activate

DEFAULT_N_IMAGES=20

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 ./run_mw_tflite.py --threads 1 --n_images $N_IMAGES
python3 ./run_mw_tflite.py --threads 4 --n_images $N_IMAGES

deactivate
