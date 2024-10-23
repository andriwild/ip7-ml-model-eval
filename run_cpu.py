from datetime import datetime
import torch
import platform
from torch import nn
from typing import cast
from torch.utils.data import DataLoader, Subset
from cpu.inference import CpuInference
from helpers.CsvWriter import CSVWriter
from helpers.arguments import parse_args

from helpers.custom_dataset import FilesystemDataset, flower_preprocessing

    #transform = transforms.Compose([
    #    transforms.Resize(640),
    #    #transforms.ToTensor(),
    #])

    # Load COCO dataset
    # dataset = datasets.CocoDetection(
    #     root=args.coco_root + '/val2017',
    #     annFile=args.coco_root + '/annotations/instances_val2017.json',
    #     transform=transform
    # )

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    # model = YoloModel('models/flower_n.pt')
    # model.eval()
    # model.to('cpu')



FLOWER_MODEL_PATH     = 'mitwelten_models/flowers_ds_v5_640_yolov5n_box_hyps_v0.pt'
POLLINATOR_MODEL_PATH = 'mitwelten_models/pollinators_ds_v6_480_yolov5s_hyps_v0.pt'
MODEL_HUB             = 'ultralytics/yolov5'
FLOWER_MODEL_DIM      = (640, 640)
POLLINATOR_MODEL_DIM  = (480, 480)
MEASUREMENT_FILE = "measurements_test"


def init_csv_writer(args) -> CSVWriter:
    header = [
            "flower_inference", 
            "pollinator_inference", 
            "n_flowers", 
            "pipeline", 
            f"meta_data: threads={args.threads}, dataset_size={args.dataset_size}, pollinator_batch_size={args.batch_size}"
            ]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return CSVWriter(f"./{MEASUREMENT_FILE}/{platform.node()}_cpu_inference_{timestamp}.csv", header)


def main() -> None:
    args = parse_args()
    n_images = args.dataset_size
    torch.set_num_threads(args.threads)

    flower_model     = cast(nn.Module, torch.hub.load(MODEL_HUB, 'custom', path=FLOWER_MODEL_PATH))
    pollinator_model = cast(nn.Module, torch.hub.load(MODEL_HUB, 'custom', path=POLLINATOR_MODEL_PATH))


    image_dataset = FilesystemDataset(
        root=r'images/root',
        #root='/home/andri/minio/images/',
        transform=lambda img: flower_preprocessing(FLOWER_MODEL_DIM, img)
    )

    # truncate dataset to the number of images specified
    subset_range = list(range(min(n_images, len(image_dataset))))
    subset_dataset = Subset(image_dataset, subset_range)

    flower_dataloader = DataLoader(
        dataset=subset_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x
    )

    writer = init_csv_writer(args)
    inference = CpuInference(flower_model, FLOWER_MODEL_DIM, pollinator_model, POLLINATOR_MODEL_DIM, writer)

    inference.run(flower_dataloader, args.batch_size)
    writer.flush()


if __name__ == "__main__":
    main()
