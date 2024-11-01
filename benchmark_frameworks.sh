#!/bin/bash

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

python3 ./run_yolo_benchmark.py --base_model yolov8n --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8n --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8n --target ncnn -n $N_IMAGES
