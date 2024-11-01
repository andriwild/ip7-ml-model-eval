from ultralytics import YOLO
import argparse
from utility.loader import load_all_images
import time
import platform
import os
import yaml
from termcolor import cprint


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
    
    for img in images:
        model.predict(img, device="cpu")
    
    end_time = time.process_time()

    cprint(f"Total time: {end_time - start_time}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {(end_time - start_time) / n_images}", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Yolo Models')
    parser.add_argument( '-m', '--base_model', type=str, default='yolov8s')
    parser.add_argument( '-t', '--target', type=str, default='tflite')
    parser.add_argument( '-n', '--n_images', type=int, default='10')
    args = parser.parse_args()

    model = args.base_model
    n_images = args.n_images
    target = args.target

    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Model file not found: {model}")

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")
    
    
    image_folder = cfg.get("image_folder")

    cprint(f"Run hailo benchmark: {model}, {n_images} images", "green")

    main(model, target, n_images, image_folder)
