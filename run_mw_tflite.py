from inference.tflite_model.tflite_inference import YoloModel
from PIL import Image
import os
import gc
from tqdm import tqdm
from datetime import datetime
import platform
import yaml
from utility.csv_writer import CSVWriter
from utility.arguments import parse_args
from time import perf_counter

with open("config.yaml", "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)


MEASUREMENT_FOLDER = cfg.get("measurement_folder")


def init_csv_writer(args) -> CSVWriter:
    header = [
            "flower_inference",
            "pollinator_inference",
            "n_flowers",
            "pipeline",
            f"meta_data: threads={args.threads}, dataset_size={args.dataset_size}, pollinator_batch_size={args.batch_size}"
            ]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return CSVWriter(f"./{MEASUREMENT_FOLDER}/{args.data_folder}/{platform.node()}_cpu_inference_{timestamp}.csv", header)


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


def load_all_images(folder_path: str) -> list[str]:
    all_images = []
    img_ext: list[str] = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in img_ext):
            img_path: str = os.path.join(folder_path, file)
            all_images.append(img_path)

    return all_images


all_images = load_all_images("/home/andri/repos/ip7-ml-model-eval/images/root/")
#all_images = load_all_images("/home/andri/fhnw/MSE/IP7/ml/dataset/flower_kaggle/flower_dataset_v4_yolo/flower_dataset_v4_yolo/images/test/")


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



def main():

    args = parse_args()
    model_1, model_2 = init_models(args.threads)
    csv_writer = init_csv_writer(args)

    for image in all_images:
        start_inference = perf_counter()
        csv_data = []
        image = Image.open(image)
    
        start_time = perf_counter()
        crops, result_class_names, result_scores = model_1.get_crops(image)
        end_time = perf_counter()
        csv_data.append(end_time - start_time)
    
        nr_flowers = len(result_class_names)

        start_time = perf_counter()
        for crop in crops:
            _ = model_2.get_crops(crop)
        end_time = perf_counter()
        csv_data.append(end_time - start_time)
        csv_data.append(nr_flowers)

    
        end_inference = perf_counter()
        csv_data.append(end_inference - start_inference)
        csv_writer.append_data(csv_data)
        csv_writer.flush()
        gc.collect()



if __name__ == "__main__":
    main()
