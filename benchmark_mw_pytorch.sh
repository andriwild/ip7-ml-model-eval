#!/usr/bin/env bash

source .venv/bin/activate

DEFAULT_N_IMAGES=20

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 ./run_mw_pytorch.py --threads 1 --n_images $N_IMAGES --batch_size 1
python3 ./run_mw_pytorch.py --threads 4 --n_images $N_IMAGES --batch_size 1

python3 ./run_mw_pytorch.py --threads 1 --n_images $N_IMAGES --batch_size 2
python3 ./run_mw_pytorch.py --threads 4 --n_images $N_IMAGES --batch_size 2

python3 ./run_mw_pytorch.py --threads 1 --n_images $N_IMAGES --batch_size 4
python3 ./run_mw_pytorch.py --threads 4 --n_images $N_IMAGES --batch_size 4

python3 ./run_mw_pytorch.py --threads 1 --n_images $N_IMAGES --batch_size 8
python3 ./run_mw_pytorch.py --threads 4 --n_images $N_IMAGES --batch_size 8

deactivate
