#!/usr/bin/env bash

set -e
source .venv/bin/activate

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

# tflite
python3 ./run_yolo_benchmark.py --base_model yolov8n --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8s --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8m --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8l --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8x --target tflite -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov5n --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5s --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5m --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5l --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5x --target tflite -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov10n --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10s --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10m --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10b --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10l --target tflite -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10x --target tflite -n $N_IMAGES


# onnx
python3 ./run_yolo_benchmark.py --base_model yolov8n --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8s --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8m --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8l --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8x --target onnx -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov5n --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5s --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5m --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5l --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5x --target onnx -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov10n --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10s --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10m --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10b --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10x --target onnx -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10l --target onnx -n $N_IMAGES

# ncnn
python3 ./run_yolo_benchmark.py --base_model yolov8n --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8s --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8m --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8l --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov8x --target ncnn -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov5n --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5s --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5m --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5l --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov5x --target ncnn -n $N_IMAGES

python3 ./run_yolo_benchmark.py --base_model yolov10n --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10s --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10m --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10b --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10x --target ncnn -n $N_IMAGES
python3 ./run_yolo_benchmark.py --base_model yolov10l --target ncnn -n $N_IMAGES


deactivate
