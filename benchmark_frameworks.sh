#!/usr/bin/env bash

source .venv/bin/activate

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

# YOLOv8
python3 ./run_yolo_benchmark.py --base_model yolov8n --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8s --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8m --target tflite -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov8n --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8s --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8m --target onnx -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov8n --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8s --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8m --target ncnn -n $N_IMAGES


# YOLOv5
python3 ./run_yolo_benchmark.py --base_model yolov5nu --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5su --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5mu --target tflite -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov5nu --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5su --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5mu --target onnx -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov5nu --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5su --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5mu --target ncnn -n $N_IMAGES

deactivate
