# curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# curl -LO http://images.cocodataset.org/zips/val2017.zip

import torch
from torch.utils.data import DataLoader

from benchmark import Measure


class CpuInference:

    def __init__(self, dataloader: DataLoader, model):
        self.dataloader = dataloader
        self.model = model


    def warm_up(self):
           for images, _labels in self.dataloader:
               x = torch.stack(images)
               out = self.model(x)
               break  # Only one batch for warm-up

    @Measure
    def run(self):
        with torch.no_grad():
            for images, _ in self.dataloader:
                x = torch.stack(images)
                _ = self.model(x)
