import fiftyone.zoo as zoo
import fiftyone

fiftyone.config.dataset_zoo_dir = "."

dataset = zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    max_samples=100,
)
