import platform
import os
import argparse
from pathlib import Path
from termcolor import cprint
import yaml
from inference.hailo.object_detection import infer
from inference.hailo.utils import load_input_images, validate_images
import csv
import gc


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
    infer(images[:batch_size], model, labels, batch_size, output_path, postprecessing=False)

    duration = infer(images, model, labels, batch_size, output_path, postprecessing=False)

    avg_inference_time = duration / n_images # in seconds
    avg_inference_time = avg_inference_time * 1000 # in milliseconds
    cprint(f"Total time: {duration} s", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {avg_inference_time} ms", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")
    return avg_inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test on Hailo8l')
    parser.add_argument( '-m', '--model', type=str, default='models/hailo8l/yolov8n.hef')
    parser.add_argument( '-b', '--batch_size', type=int, default='10')
    parser.add_argument( '-n', '--n_images', type=int, default='10')
    parser.add_argument( '-l', '--labels', type=str, default='coco/coco.txt')
    parser.add_argument( '-t', '--target', type=str)
    args = parser.parse_args()

    model_path = args.model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
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

    cprint(f"Run hailo benchmark: {args.model}, batch_size={batch_size}, {n_images} images, labels:{labels}", "green")
    avg_inference_time = main(model_path, batch_size, n_images, labels, image_folder)

    model_name = model_path.split("/")[-1].split(".")[0]

    write_results([platform.node(), args.target, model_name, "hailo", avg_inference_time, batch_size], output_file)
    gc.collect()
