import argparse
import numpy as np
import sys
import time
import yaml
import gc
import platform
import csv
from tflite_runtime.interpreter import Interpreter
import tflite_runtime.interpreter as tflite
from PIL import Image
from termcolor import cprint
from utility.loader import load_all_images
from multiprocessing import cpu_count


def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")



def main(model_path, num_images, image_folder):
    interpreter = Interpreter(
            model_path=model_path,
            num_threads=cpu_count(),
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
            )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    height, width = input_details[0]['shape'][1:3]
    input_dtype = input_details[0]['dtype']
    input_scale, input_zero_point = input_details[0]['quantization']

    image_paths = load_all_images(image_folder, num_images)
    if not image_paths:
        print(f"No images found in {args.image_dir}!")
        sys.exit()

    inference_time = 0
    for image_path in image_paths:

        image = Image.open(image_path).convert('RGB')

        image_resized = image.resize((width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if input_dtype == np.float32:
            # Floating Point Modell
            input_data = (np.float32(input_data) / 255.0)
        elif input_dtype == np.uint8:
            # uint8 quant modell
            if input_scale != 0:
                input_data = np.uint8((np.float32(input_data) / 255.0) / input_scale + input_zero_point)
            else:
                input_data = np.uint8(input_data)
        elif input_dtype == np.int8:
            # int8 quant modell
            if input_scale != 0:
                input_data = np.int8((np.float32(input_data) / 255.0 - input_zero_point) / input_scale)
            else:
                input_data = np.int8(input_data - 128)
        else:
            raise ValueError(f"Unsupported input type: {input_dtype}")


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
    parser.add_argument('--model_path', required=True, help='Pfad zur Modelldatei')
    parser.add_argument('--n_images', type=int, default=10, help='Anzahl der zu verarbeitenden Bilder')
    args = parser.parse_args()

    n_images = args.n_images
    model_path = args.model_path

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")

    image_folder = cfg.get("image_folder")
    output_file = cfg.get("output_file")

    cprint(f"Run beagle benchmark: {model_path}, {n_images} images", "green")
    avg_inference_time = main(model_path, n_images, image_folder)
    model_name = model_path.split("/")[-1].split(".")[0]

    write_results([model_name, "m2 edgetpu", avg_inference_time], output_file)
    gc.collect()

