#!/bin/bash

echo "Make sure you have activated the hailo virtual environment before running this script."

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 run_hailo.py --model_path models/hailo/yolov8n.hef --batch_size 10 --dataset_size $N_IMAGES
