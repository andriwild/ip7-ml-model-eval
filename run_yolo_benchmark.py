from ultralytics import YOLO
import argparse
from utility.loader import load_all_images
import time


def main(base_model, target, n_images):
    model = YOLO(base_model)
    model_path = model.export(format=target)

    target_model = YOLO(model_path)

    image_folder = "coco/val2017"
    images = load_all_images(image_folder)
    images = images[:n_images]
    
    # Warm up the model
    target_model.predict(images[0], device="cpu")
    # Run inference

    # Start the benchmark
    start_time = time.process_time()
    
    for img in images:
        model.predict(img, device="cpu")
    
    end_time = time.process_time()

    # Calculate performance metrics
    total_time = end_time - start_time
    avg_time_per_image = total_time / n_images 
    
    print(f"Total inference time for {n_images} images: {total_time:.2f} seconds")
    print(f"Average inference time per image: {avg_time_per_image:.4f} seconds per image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test: Yolo Models')
    parser.add_argument( '-m', '--base_model', type=str, default='yolov8s')
    parser.add_argument( '-t', '--target', type=str, default='tflite')
    parser.add_argument( '-n', '--n_images', type=int, default='1000')
    args = parser.parse_args()

    main(args.base_model, args.target, args.n_images)
