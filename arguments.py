import argparse
import os

YOLO_MODELS = [
        "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
        "yolov8n.hef", "yolov8s.hef", "yolov8m.hef", "yolov8l.hef", "yolov8x.hef",
        ]

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
            default='models/yolov8n.pt', 
            help='Model name (e.g., /models/[yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, ...])', # TODO: Update the list of models
            #choices=[*YOLO_MODELS, "all"]
            )

    parser.add_argument(
            '--coco_root',
            type=str,
            default='./coco',
            help='Path to the COCO dataset root directory'
            )

    parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='Device to use (cuda or cpu)'
            )

    parser.add_argument(
            '-i', '--iterations',
            type=int,
            default=10,
            help='Number of iterations for measurement'
            )

    parser.add_argument(
            '-b', '--batch_size',
            type=int,
            default=1,
            help="Number of images in one batch"
            )

    parser.add_argument(
            '-d', '--debug',
            action='store_true',
            default=False, help='Show sample images during inference'
            )
    parser.add_argument(
        "-l", "--labels", 
        default="coco/coco.txt", 
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )

    args = parser.parse_args()

    # Validate paths
    # if not os.path.exists(args.model_path):
    #     raise FileNotFoundError(f"Network file not found: {args.model_path}")
    # if not os.path.exists(args.coco_root):
    #     raise FileNotFoundError(f"Input path not found: {args.coco_root}")
    # if not os.path.exists(args.labels):
    #     raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args
