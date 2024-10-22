import argparse
import os

from multiprocessing import cpu_count


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description='ML Model Inference Time Measurement')

    # Arguments
    parser.add_argument(
            '-m', '--model_path',  
            type=str,  
            default='./yolov5s.pt',
            help='Model name (e.g., /models/yolov8n.pt)'
            )
    parser.add_argument(
            '--coco_root',
            type=str,
            default='./coco',
            help='Path to the COCO dataset root directory'
            )
    parser.add_argument(
            '-t', '--threads',
            type=int,
            default=cpu_count(),
            help='Number of threads'
            )

    parser.add_argument(
            '-d','--device',
            type=str,
            default='cpu',
            help='Device to use (cuda or cpu)'
            )

    parser.add_argument(
            '-s', '--dataset_size',
            type=int,
            default=10,
            help='Number of images to use for inference'
            )

    parser.add_argument(
            '-b', '--batch_size',
            type=int,
            default=8,
            help="Number of images in one batch"
            )

    parser.add_argument(
            '--debug',
            action='store_true',
            default=False, help='Show sample images during inference'
            )
    parser.add_argument(
        "-l", "--labels", 
        default="./coco/coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.coco_root):
        raise FileNotFoundError(f"Input path not found: {args.coco_root}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args
