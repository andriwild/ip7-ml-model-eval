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


def main(model, target_name, target_variant, n_images, image_folder):

    model = YOLO(model)
    if target_name != "pt":

        if target_variant == "int8":
            model_path = model.export(format=target_name, int8=True)
        elif target_variant == "f16":
            model_path = model.export(format=target_name, half=True)
        else:
            model_path = model.export(format=target_name)

        model = YOLO(model_path)

    images = load_all_images(image_folder, n_images)
    
    # Warm up
    model.predict(images[0], device="cpu")

    # Start the benchmark
    inference_time = 0
    for image in images:
        output = model.predict(image, device="cpu")
        inference_time += output[0].speed["inference"]

    avg_inference_time = inference_time / n_images
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {avg_inference_time}", "green")
    cprint(f"Inference runs on: {platform.uname()}", "green")
    return avg_inference_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Yolo Models')
    parser.add_argument( '-m', '--model', type=str, default='yolov8s')
    parser.add_argument( '-t', '--type', type=str, default='pt')
    parser.add_argument( '-n', '--n_images', type=int, default='10')
    args = parser.parse_args()

    model = args.model
    n_images = args.n_images
    target = args.type

    target_parts = target.split("_")
    if len(target_parts) == 2:
        target_name = target_parts[0]
        target_variant = target_parts[1]
    else:
        target_name = target
        target_variant = None

    with open("config.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise FileNotFoundError(f"Config file not found")
    
    
    image_folder = cfg.get("image_folder")
    output_file = cfg.get("output_file")

    cprint(f"Run yolo benchmark: {model} {target_name}_{target_variant}, {n_images} images", "green")


    avg_inference_time = main(model, target_name, target_variant, n_images, image_folder)

    write_results([platform.node(), "no accelerator", model, f"{target_name}_{target_variant}", avg_inference_time, 1], output_file)
    gc.collect()

