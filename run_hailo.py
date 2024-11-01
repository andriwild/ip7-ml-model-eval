import platform
import time
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

    
    # warm up
    infer(images[:2], args.model_path, args.labels, args.batch_size, output_path, postprecessing=False)

    start = time.process_time()
    infer(images, args.model_path, args.labels, args.batch_size, output_path, postprecessing=False)
    end = time.process_time()
    cprint(f"Total time: {end - start}", "green")
    cprint(f"Number of processed images: {n_images}", "green")
    cprint(f"Average time per image: {(end - start) / n_images}", "green")


if __name__ == "__main__":
    main()
