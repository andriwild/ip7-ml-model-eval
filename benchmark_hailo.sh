#!/usr/bin/env bash

set -e
source .tpu/bin/activate

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 run_hailo.py --model models/hailo/yolov8n.hef --batch_size 10 --n_images $N_IMAGES

deaactivate
