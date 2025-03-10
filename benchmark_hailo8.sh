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
    yolov5m
    yolov5s
    yolov8l
    yolov8m
    yolov8n
    yolov8s
    yolov8x
)


for MODEL in "${MODELS[@]}"; do
    python3 run_hailo.py --model "models/hailo8/${MODEL}.hef" --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES --target "hailo8"
    sleep 2 
done

deactivate
