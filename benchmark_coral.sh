#!/bin/bash

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 ./run_coral_usb.py --model yolov8s --n_images $N_IMAGES
python3 ./run_coral_usb.py --model yolov8n --n_images $N_IMAGES
python3 ./run_coral_usb.py --model yolov11n --n_images $N_IMAGES
