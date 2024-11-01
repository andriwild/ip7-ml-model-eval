import time
import platform
import argparse
from ultralytics import YOLO
import os
import yaml
from utility.loader import load_all_images
from termcolor import cprint


def main(model_name, n_images, image_folder):
    # Load the model
    model = YOLO(f"edgetpu/{model_name}_full_integer_quant_edgetpu.tflite")
    
    images = load_all_images(image_folder)
    images = images[:n_images]

    # Warm up the model
    model.predict(images[0], device="tpu:0")
    
    
    # Start the benchmark
    start_time = time.process_time()
    
    for img in images:
        model.predict(img, device="tpu:0")
    
    end_time = time.process_time()

    cprint(f"Total time: {end_time - start_time}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {(end_time - start_time) / n_images}", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Coral USB Accelerator')
    parser.add_argument( '-m', '--model', type=str, default='yolov8s')
    parser.add_argument( '-n', '--n_images', type=int, default='1000')
    args = parser.parse_args()

    model = args.model
    n_images = args.n_images

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")
    
    
    image_folder = cfg.get("image_folder")

    cprint(f"Run hailo benchmark: {model}, {n_images} images", "green")
    main(model, n_images, image_folder)
