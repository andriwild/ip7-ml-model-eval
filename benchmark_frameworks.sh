#!/usr/bin/env bash

source .venv/bin/activate

DEFAULT_N_IMAGES=10

N_IMAGES=${1:-$DEFAULT_N_IMAGES}

models_v8=("yolov8n" "yolov8s" "yolov8m")
models_v5=("yolov5nu" "yolov5su" "yolov5mu")
types=("tflite" "onnx" "ncnn" "pt")

# YOLOv8
for model in "${models_v8[@]}"; do
  for type in "${types[@]}"; do
    python3 ./run_yolo_benchmark.py --model "$model" --type "$type" -n "$N_IMAGES"
  done
done

# YOLOv5
for model in "${models_v5[@]}"; do
  for type in "${types[@]}"; do
    python3 ./run_yolo_benchmark.py --model "$model" --type "$type" -n "$N_IMAGES"
  done
done

deactivate
