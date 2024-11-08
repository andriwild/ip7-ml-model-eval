#!/usr/bin/env bash

source .hailo/bin/activate

DEFAULT_N_IMAGES=10
DEFAULT_BATCH_SIZE=1

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

MODELS=(
    yolov10b
    yolov10n
    yolov10s
    yolov10x
    yolov5mu
    yolov5su
    yolov8l
    yolov8m
    yolov8n
    yolov8s
    yolov8x
)

# 8l Models

for MODEL in "${MODELS[@]}"; do
    python3 run_hailo.py --model "models/hailo8l/${MODEL}.hef" --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
    sleep 2 
done

# 8 Models

for MODEL in "${MODELS[@]}"; do
    python3 run_hailo.py --model "models/hailo8/${MODEL}.hef" --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
    sleep 2 
done

deactivate
