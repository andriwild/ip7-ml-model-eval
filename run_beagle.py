import os
import argparse
import numpy as np
import sys
import time
import csv
from typing import List
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw
from termcolor import cprint  # Falls für farbige Ausgaben benötigt
import yaml
import gc
import platform


def load_all_images(folder_path: str, num_images: int) -> List[str]:
    images = []
    img_ext = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in img_ext):
            img_path = os.path.join(folder_path, file)
            images.append(img_path)
            if len(images) >= num_images:
                break

    return images


def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")



def main(model_path, num_images, image_dir):

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    height, width = input_details[0]['shape'][1:3]
    floating_model = (input_details[0]['dtype'] == np.float32)

    image_paths = load_all_images(args.image_dir, num_images)
    if not image_paths:
        print(f"Keine Bilder im Verzeichnis {args.image_dir} gefunden.")
        sys.exit()

    inference_time = 0
    for image_path in image_paths:

        image = Image.open(image_path).convert('RGB')

        image_resized = image.resize((width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) / 255.0)
        else:
            input_data = np.uint8(input_data)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        inference_time += end_time - start_time

    avg_inference_time = inference_time / n_images
    cprint(f"Total time: {inference_time}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {avg_inference_time}", "green")
    cprint(f"Inference runs on: {platform.node()}", "green")
    return avg_inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Pfad zur .tflite-Modelldatei')
    parser.add_argument('--image_dir', required=True, help='Verzeichnis mit den Bildern')
    parser.add_argument('--n_images', type=int, default=10, help='Anzahl der zu verarbeitenden Bilder')
    args = parser.parse_args()

    model = args.model
    n_images = args.num_images
    image_dir = args.image_dir
    model_path = f"models/edgetpu/{model}.tflite"

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")

    image_folder = cfg.get("image_folder")
    output_file = cfg.get("output_file")

    cprint(f"Run beagle benchmark: {model_path}, {n_images} images", "green")
    avg_inference_time = main(model_path, n_images, image_dir)

    write_results([model, "coral usb", avg_inference_time], output_file)
    gc.collect()

