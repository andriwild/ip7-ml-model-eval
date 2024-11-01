import platform
import os
import argparse
import time
from pathlib import Path
from termcolor import cprint
import yaml
from inference.hailo.object_detection import infer
from inference.hailo.utils import load_input_images, validate_images
import csv


def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")


def main(model, batch_size, n_images, labels, image_folder) -> None:

    # Load input images
    images = load_input_images(image_folder)
    images = images[:n_images]

    # Validate images
    try:
        validate_images(images, batch_size)
    except ValueError as e:
        cprint(e, "red")
        return

    # Create output directory if it doesn't exist
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)
    
    # warm up
    infer(images[:2], model, labels, batch_size, output_path, postprecessing=False)

    start_time = time.time()
    infer(images, model, labels, batch_size, output_path, postprecessing=False)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / n_images
    cprint(f"Total time: {end_time - start_time}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {avg_inference_time}", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")
    return avg_inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test on Hailo8l')
    parser.add_argument( '-m', '--model', type=str, default='models/hailo/yolov8n.hef')
    parser.add_argument( '-b', '--batch_size', type=int, default='10')
    parser.add_argument( '-n', '--n_images', type=int, default='10')
    parser.add_argument( '-l', '--labels', type=str, default='coco/coco.txt')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    model = args.model
    batch_size = args.batch_size
    n_images = args.n_images
    labels = args.labels


    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")
    
    
    image_folder = cfg.get("image_folder")
    output_file = cfg.get("output_file")

    cprint(f"Run hailo benchmark: {model}, batch_size={batch_size}, {n_images} images, labels:{labels}", "green")
    avg_inference_time = main(model, batch_size, n_images, labels, image_folder)

    write_results([model, batch_size, avg_inference_time], output_file)
