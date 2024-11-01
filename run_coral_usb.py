import time
import argparse
from ultralytics import YOLO
import os
from utility.loader import load_all_images


def main(model_name, n_images):
    # Load the model
    model = YOLO(f"edgetpu/{model_name}_full_integer_quant_edgetpu.tflite")
    
    image_folder = "/coco/val2017"
    images = load_all_images(image_folder)
    images = images[:n_images]

    # Warm up the model
    model.predict(images[0], device="tpu:0")
    
    
    # Start the benchmark
    start_time = time.process_time()
    
    for img in images:
        model.predict(img, device="tpu:0")
    
    end_time = time.process_time()
    
    # Calculate performance metrics
    total_time = end_time - start_time
    avg_time_per_image = total_time / n_images 
    
    print(f"Total inference time for {n_images} images: {total_time:.2f} seconds")
    print(f"Average inference time per image: {avg_time_per_image:.4f} seconds per image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Coral USB Accelerator')
    parser.add_argument( '-m', '--model', type=str, default='yolov8s')
    parser.add_argument( '-n', '--n_images', type=int, default='1000')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    main(args.model, args.n_images)
