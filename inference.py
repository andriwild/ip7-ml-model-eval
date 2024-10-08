# curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# curl -LO http://images.cocodataset.org/zips/val2017.zip

import platform
import time

from arguments import parse_args
from ultralytics import YOLO
from termcolor import cprint
import torch
import logging
from csv_logger import CsvLogger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from hailo.object_detection import infer
from hailo.utils import load_input_images, validate_images

time_of_execution=datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

filename = f'reports/{time_of_execution}-{platform.node()}-inference-test.csv'
delimiter = ','
level = logging.INFO
fmt =f'%(asctime)s,%(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'
max_size = 1024  # 1 kilobyte
max_files = 4  # 4 rotating files
header = ['date', 'device', 'model', 'time_ms', 'iterations']

# Create logger with csv rotating handler
csvlogger = CsvLogger(filename=filename,
                      delimiter=delimiter,
                      level=level,
                      add_level_nums=None,
                      fmt=fmt,
                      datefmt=datefmt,
                      max_size=max_size,
                      max_files=max_files,
                      header=header)

YOLO_MODELS = [
        "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
        ]

def measure_inference_time(
    model: YOLO,
    dataloader: DataLoader,
    device: torch.device,
    iterations: int = 100,
    debug: bool = False
) -> float:
    # Set the device
    model.to(device)

    # Warm-up
    for images, _ in dataloader:
        images = images.to(device)
        _ = model(images)
        break  # Only one batch for warm-up

    # Inference time measurement start
    start_time: float = time.perf_counter()

    with torch.no_grad():
        count: int = 0
        for images, _ in dataloader:
            images = images.to(device)
            # run batches inference
            result = model(images)

            if debug and count % 10 == 0:
                result[0].show()

            count += 1
            if count >= iterations:
                break

    # Inference time measurement end
    end_time: float = time.perf_counter()

    total_time: float = end_time - start_time
    avg_time_per_inference: float = total_time / iterations
    return avg_time_per_inference


def run_hailo_inference(args) -> None:
    # Load input images
    images = load_input_images(args.input)
    
    # Validate images
    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        cprint(e, "red")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)

    # Start the inference
    infer(images, args.net, args.labels, args.batch_size, output_path)



def main() -> None:
    args = parse_args()

    hailo_model = args.model_path.endswith(".hef")
    if hailo_model:
        cprint("Running on Hailo Accelerator", "red")
        run_hailo_inference(args)
        exit(1)

    # Load COCO dataset
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    coco_dataset = datasets.CocoDetection(
        root=args.coco_root + '/val2017',
        annFile=args.coco_root + '/annotations/instances_val2017.json',
        transform=transform
    )
    dataloader = DataLoader(coco_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    if args.model_path == "all":
        selected_models = YOLO_MODELS
    else:
        selected_models = [args.model_path]

    for model_path in tqdm(selected_models):
        model = YOLO(model_path)
        avg_time = measure_inference_time(model, dataloader, device, args.iterations, args.debug)

        csvlogger.info(msg=[platform.node(), model_path, avg_time, args.iterations])
        cprint(f'Average inference time for {model_path} on {platform.node()} {device}: {avg_time * 1000:.2f} ms', "blue")


    if args.debug:
        all_logs = csvlogger.get_logs(evaluate=False)
        for log in all_logs:
            print(log)
    exit(0)


if __name__ == "__main__":
    main()
