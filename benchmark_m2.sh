#!/usr/bin/env bash

source .m2/bin/activate

DEFAULT_N_IMAGES=10

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

for MODEL in "${MODELS[@]}"; do
    python3 ./run_m2_edgetpu.py --model "models/edgetpu/${MODEL}_full_integer_quant_edgetpu.tflite" --n_images $N_IMAGES
    sleep 1 
done

deactivate
