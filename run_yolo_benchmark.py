from ultralytics import YOLO
import argparse
from utility.loader import load_all_images
import time
import platform
import os
import yaml
from termcolor import cprint
import csv


def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")


def main(base_model, target, n_images, image_folder):
    model = YOLO(base_model)
    model_path = model.export(format=target)

    target_model = YOLO(model_path)

    images = load_all_images(image_folder)
    images = images[:n_images]
    
    # Warm up
    target_model.predict(images[0], device="cpu")

    # Start the benchmark
    start_time = time.process_time()
    for image in images:
        target_model.predict(image, device="cpu")
    end_time = time.process_time()

    avg_inference_time = (end_time - start_time) / n_images
    cprint(f"Total time: {end_time - start_time}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {avg_inference_time}", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")
    return avg_inference_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Yolo Models')
    parser.add_argument( '-m', '--base_model', type=str, default='yolov8s')
    parser.add_argument( '-t', '--target', type=str, default='tflite')
    parser.add_argument( '-n', '--n_images', type=int, default='10')
    args = parser.parse_args()

    model = args.base_model
    n_images = args.n_images
    target = args.target

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")
    
    
    image_folder = cfg.get("image_folder")
    output_file = cfg.get("output_file")

    cprint(f"Run hailo benchmark: {model}, {n_images} images", "green")

    avg_inference_time = main(model, target, n_images, image_folder)

    write_results([model, target, avg_inference_time], output_file)

