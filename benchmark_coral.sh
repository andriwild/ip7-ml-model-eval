#!/usr/bin/env bash

source .tpu/bin/activate

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 ./run_coral_usb.py --model yolov8s --n_images $N_IMAGES

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

for MODEL in "${MODELS[@]}"; do
    python3 ./run_coral_usb.py --model "$MODEL" --n_images $N_IMAGES
    sleep 1 
done

deactivate
