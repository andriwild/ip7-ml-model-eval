import platform
import argparse
from ultralytics import YOLO
import os
import yaml
from utility.loader import load_all_images
from termcolor import cprint
import csv
import gc


def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")


def main(model_path, n_images, image_folder):
    # Load the model
    model = YOLO(model_path)
    
    images = load_all_images(image_folder, n_images)

    # Warm up the model
    model.predict(images[0], device="tpu:0")
    
    # Start the benchmark
    inference_time = 0
    print(len(images))
    for image in images:
        output = model.predict(image, device="tpu:0")
        inference_time += output[0].speed["inference"]

    avg_inference_time = inference_time / n_images
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {avg_inference_time}", "green")
    cprint(f"Inference runs on: {platform.uname()}", "green")
    return avg_inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Coral USB Accelerator')
    parser.add_argument( '-m', '--model', type=str, default='yolov8s')
    parser.add_argument( '-n', '--n_images', type=int, default='10')
    args = parser.parse_args()

    model = args.model
    n_images = args.n_images

    model_path = f"models/edgetpu/{model}_full_integer_quant_edgetpu.tflite"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")
    
    
    image_folder = cfg.get("image_folder")
    output_file = cfg.get("output_file")

    cprint(f"Run hailo benchmark: {model}, {n_images} images", "green")
    avg_inference_time = main(model_path, n_images, image_folder)

    write_results([platform.node(), "usb_edgetpu", model, "tflite", avg_inference_time, 1], output_file)
    gc.collect()
