#!/usr/bin/env bash

source .venv/bin/activate

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

# YOLOv8
python3 ./run_yolo_benchmark.py --model yolov8n --type tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8s --type tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8m --type tflite -n $N_IMAGES

python3 ./run_yolo_benchmark.py --model yolov8n --type onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8s --type onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8m --type onnx -n $N_IMAGES

python3 ./run_yolo_benchmark.py --model yolov8n --type ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8s --type ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8m --type ncnn -n $N_IMAGES

python3 ./run_yolo_benchmark.py --model yolov8n --type pt -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8s --type pt -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov8m --type pt -n $N_IMAGES


# YOLOv5
python3 ./run_yolo_benchmark.py --model yolov5nu --type tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5su --type tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5mu --type tflite -n $N_IMAGES

python3 ./run_yolo_benchmark.py --model yolov5nu --type onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5su --type onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5mu --type onnx -n $N_IMAGES

python3 ./run_yolo_benchmark.py --model yolov5nu --type ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5su --type ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5mu --type ncnn -n $N_IMAGES

python3 ./run_yolo_benchmark.py --model yolov5nu --type pt -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5su --type pt -n $N_IMAGES
python3 ./run_yolo_benchmark.py --model yolov5mu --type pt -n $N_IMAGES

deactivate
