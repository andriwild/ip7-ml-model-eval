import platform
from pathlib import Path

from termcolor import cprint

from utility.arguments import parse_args
from inference.hailo.object_detection import infer
from inference.hailo.utils import load_input_images, validate_images


def main() -> None:

    args = parse_args()
    n_images = args.dataset_size

    # Load input images
    images = load_input_images("coco/images/val2017/")
    images = images[:n_images]

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
    print("run inference: ", args.model_path, args.labels, args.batch_size, output_path)
    infer(images, args.model_path, args.labels, args.batch_size, output_path)
    print(duration / n_images)
    print(platform.node(), args.device, "yolo", duration / n_images, n_images, "coco")


if __name__ == "__main__":
    main()
