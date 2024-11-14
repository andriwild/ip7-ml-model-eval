from datetime import datetime
import torch
import platform
import argparse
from termcolor import cprint
from torch import nn
from typing import cast
from torch.utils.data import DataLoader, Subset
from inference.pytorch_model.pytorch_inference import CpuInference
from utility.csv_writer import CSVWriter
from utility.custom_dataset import preprocessing
from utility.custom_dataset import FilesystemDataset


FLOWER_MODEL_PATH     = 'models/mitwelten_models/flowers_ds_v5_640_yolov5n_box_hyps_v0.pt'
POLLINATOR_MODEL_PATH = 'models/mitwelten_models/pollinators_ds_v6_480_yolov5s_hyps_v0.pt'
MODEL_HUB             = 'ultralytics/yolov5'
FLOWER_MODEL_DIM      = (640, 640)
POLLINATOR_MODEL_DIM  = (480, 480)


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
    return CSVWriter(f"./data/mw_pytorch/{platform.node()}_cpu_inference_{timestamp}.csv", header)


def main(n_images, batch_size, threads) -> None:
    flower_model     = cast(nn.Module, torch.hub.load(MODEL_HUB, 'custom', path=FLOWER_MODEL_PATH))
    pollinator_model = cast(nn.Module, torch.hub.load(MODEL_HUB, 'custom', path=POLLINATOR_MODEL_PATH))


    image_dataset = FilesystemDataset(
        root='images/root/',
        transform=lambda img: preprocessing(FLOWER_MODEL_DIM, img)
    )

    # truncate dataset to the number of images specified
    subset_range = list(range(min(n_images, len(image_dataset))))
    subset_dataset = Subset(image_dataset, subset_range)
    cprint(f"Run mitwelten benchmark: {n_images} images, batch_size: {batch_size}, threads: {threads}", "green")

    flower_dataloader = DataLoader(
        dataset=subset_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x
    )

    writer = init_csv_writer(args)

    inference = CpuInference(flower_model, FLOWER_MODEL_DIM, pollinator_model, POLLINATOR_MODEL_DIM, writer)
    inference.run(flower_dataloader, batch_size)

    cprint("Inference done", "green")
    writer.append_data(threads)
    writer.append_data(batch_size)
    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=10, help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--threads', type=int, default=1, help='Threads to use for the inference')
    args = parser.parse_args()

    n_images = args.n_images
    torch.set_num_threads(args.threads)

    main(n_images, args.batch_size, args.threads)

