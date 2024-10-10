import platform

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from arguments import parse_args
from cpu.inference import CpuInference


def main() -> None:
    args = parse_args()
    n_images = args.dataset_size

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    # Load COCO dataset
    coco_dataset = datasets.CocoDetection(
        root=args.coco_root + '/val2017',
        annFile=args.coco_root + '/annotations/instances_val2017.json',
        transform=transform
    )

    subset_range = list(range(min(n_images, len(coco_dataset))))
    subset_dataset = Subset(coco_dataset, subset_range)

    dataloader = DataLoader(
        dataset=subset_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    torch.set_num_threads(args.cpu)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #model = torch.load("yolov5s.pt")
    model.eval()
    model.to('cpu')

    i = CpuInference(dataloader, model)

    i.warm_up()
    duration = i.run()
    print(duration / n_images)
    print(platform.node(), args.device, "yolo", duration / n_images, n_images, "coco")


if __name__ == "__main__":
    main()
