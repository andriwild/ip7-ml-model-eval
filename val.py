from ultralytics import YOLO
import argparse
from utility.loader import load_all_images
import platform
import yaml
from termcolor import cprint
import csv
import gc


def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")

# Load a pre-trained YOLO model

# yolov8n,tflite_f32,37,03559118251259
# yolov8s,onnx,44,71588638716728
# yolov8s,ncnn,44,71588640173783
# yolov8s,tflite_f32,44.71590925643171
# yolov8s,tflite_int8,44.23437499989735
# yolov8s,tflite_f16,44.71548035739889
# yolov8m,onnx,49.993172185209817
# yolov8m,ncnn,49.99317267328455

models = ["yolov8n", "yolov5su", "yolov5nu", "yolov5mu", "yolov10b", "yolov10n", "yolov10s"]
formats = ["onnx", "ncnn", "tflite_int8"]

for model in models:
    cprint(f"Model: {model}", "green")
    yolo_model = YOLO(f"{model}.pt")

    for format in formats:
        cprint(f"Format: {model} {format}", "green")
        target_parts = format.split("_")

        if len(target_parts) == 2:
            target_name = target_parts[0]
            target_variant = target_parts[1]
        else:
            target_name = format
            target_variant = None

        if target_variant == "int8":
            cprint(f"int8=True", "green")
            model_path = yolo_model.export(format=target_name, int8=True)
        elif target_variant == "f16":
            cprint(f"half=True", "green")
            model_path = yolo_model.export(format=target_name, half=True)
        else:
            model_path = yolo_model.export(format=target_name)

        cprint(f"Model exported to: {model_path}", "green")
        yolo_model2 = YOLO(model_path)
        metrics = yolo_model2.val(data="coco.yaml")

        cprint(metrics.box.map, "red")

        print_target = target_name
        if target_variant:
            print_target += f"_{target_variant}"

        write_results([model, print_target, str(metrics.box.map * 100)], "data/map.csv")

# yolov8n_f32 = 0.3703559118251259

