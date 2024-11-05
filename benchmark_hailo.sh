#!/usr/bin/env bash

source .hailo/bin/activate

DEFAULT_N_IMAGES=10
DEFAULT_BATCH_SIZE=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 run_hailo.py --model yolov10b --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES 
sleep 2
python3 run_hailo.py --model yolov10n --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov10s --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov10x --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2

python3 run_hailo.py --model yolov5m  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov5s  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES

python3 run_hailo.py --model yolov8l  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov8m  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov8n  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov8s  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2
python3 run_hailo.py --model yolov8x  --batch_size $DEFAULT_BATCH_SIZE --n_images $N_IMAGES
sleep 2

deactivate
