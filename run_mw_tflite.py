from inference.tflite_model.tflite_inference import YoloModel
from PIL import Image
import gc
from datetime import datetime
import torch
import platform
import argparse
from termcolor import cprint
from utility.csv_writer import CSVWriter
from time import perf_counter
from utility.loader import load_all_images


def init_csv_writer(args) -> CSVWriter:
    header = [
            "flower_inference",
            "pollinator_inference",
            "n_flowers",
            "pipeline",
            "threads",
            "batch_size",
            ]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return CSVWriter(f"./data/mw_tflite/{platform.node()}_cpu_inference_{timestamp}.csv", header)



def draw_boxes(draw, boxes, labels, scores, det, cropped_image):
    ymin, xmin, ymax, xmax = det['box']
    # Convert normalized coordinates to pixel values
    left = int(xmin * cropped_image.width)
    top = int(ymin * cropped_image.height)
    right = int(xmax * cropped_image.width)
    bottom = int(ymax * cropped_image.height)

    # Draw the bounding box
    draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

    # Draw the label
    label = f"Class {det['class']}: {det['score']:.2f}"
    draw.text((left, top - 10), label, fill="red")



def init_models(n_threads):
    model_1 = YoloModel(
        "models/mitwelten_models/flowers_ds_v5_640_yolov5n_v0_cnv-fp16.tflite",
        640,
        0.5,
        0.5,
        classes=['daisy', 'wildemoere', 'flockenblume'],
        margin=20,
        n_threads=n_threads
    )
    model_2 = YoloModel(
        "models/mitwelten_models/pollinators_ds_v6_480_yolov5s_bs32_300ep_multiscale_v0-fp16.tflite",
        480,
        0.8,
        0.5,
        classes=["honigbiene", "wildbiene","hummel","schwebfliege","fliege"],
        margin=20,
        n_threads=n_threads
    )
    return model_1, model_2



def main(n_images, threads):

    all_images = load_all_images("images/root/", n_images)
    model_1, model_2 = init_models(threads)
    csv_writer = init_csv_writer(args)
    cprint(f"Run mitwelten benchmark: {n_images} images", "green")

    for image in all_images:
        start_inference = perf_counter()
        image = Image.open(image)
    
        start_time = perf_counter()
        crops, result_class_names, _result_scores = model_1.get_crops(image)
        end_time = perf_counter()
        csv_data = [(end_time - start_time)]
    
        nr_flowers = len(result_class_names)

        start_time = perf_counter()
        for crop in crops:
            _ = model_2.get_crops(crop)
        end_time = perf_counter()
        csv_data.append(end_time - start_time)
        csv_data.append(nr_flowers)

    
        end_inference = perf_counter()
        csv_data.append(end_inference - start_inference)
        csv_data.append(threads)
        csv_data.append(1)
        csv_writer.append_data(csv_data)
        csv_writer.flush()
        cprint(f"Image processed", "green")
        gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=10, help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--threads', type=int, default=1, help='Threads to use for the inference')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    main(args.n_images, args.threads)
