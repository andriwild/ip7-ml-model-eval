import platform
import argparse
import time
from pathlib import Path
from termcolor import cprint
import yaml
from utility.arguments import parse_args
from inference.hailo.object_detection import infer
from inference.hailo.utils import load_input_images, validate_images


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

    start = time.process_time()
    infer(images, model, labels, batch_size, output_path, postprecessing=False)
    end = time.process_time()
    cprint(f"Total time: {end - start}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {(end - start) / n_images}", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Yolo Models')
    parser.add_argument( '-m', '--model', type=str, default='models/hailo/yolov8n.hef')
    parser.add_argument( '-b', '--batch_size', type=int, default='10')
    parser.add_argument( '-n', '--n_images', type=int, default='100')
    parser.add_argument( '-l', '--labels', type=str, default='coco/coco.txt')
    args = parser.parse_args()
    
    model = args.model
    batch_size = args.batch_size
    n_images = args.n_images
    labels = args.labels

    cprint(f"Run hailo benchmark: {model}, batch_size={batch_size}, {n_images} images, labels:{labels}", "green")

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    
    
    image_folder = cfg.get("image_folder")

    main(model, batch_size, n_images, labels, image_folder)
