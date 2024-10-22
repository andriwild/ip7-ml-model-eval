# curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# curl -LO http://images.cocodataset.org/zips/val2017.zip

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np
from helpers.benchmark import TimeMeasure
from typing import List
from time import perf_counter

@dataclass
class DetectionBox:
    x_max: float
    y_max: float
    x_min: float
    y_min: float

    def __repr__(self):
        return f"Box(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"

    def to_tuple(self):
        # PIL conform box tuple
        return (self.x_min, self.y_min, self.x_max, self.y_max) 
    
    def scale(self, scale_x, scale_y) -> 'DetectionBox':
       return DetectionBox(
           self.x_min * scale_x,
           self.y_min * scale_y,
           self.x_max * scale_x,
           self.y_max * scale_y
           )

    def add_margin(self, margin: int, max_w, max_h) -> 'DetectionBox':
        # add margin to the box, if the margin exceeds the image size, clip it
        return DetectionBox(
                np.clip(self.x_min - margin, 0, max_w),
                np.clip(self.y_min - margin, 0, max_h),
                np.clip(self.x_max + margin, 0, max_w),
                np.clip(self.y_max + margin, 0, max_h),
            )



@dataclass
class CpuInference:

    flower_model: nn.Module
    flower_model_dim: tuple[int, int]
    pollinator_model: nn.Module
    pollinator_model_dim: tuple[int, int]

    @TimeMeasure(name="flower_inference")
    def _run_inference(self, images):
        return self.flower_model(images)

    @TimeMeasure(name="flower_postprocessing")
    def _postprocessing(self, detections, image_specs) -> List[Image.Image]:

        cropped_images = []

        for (det, spec) in zip(detections.tolist(), image_specs):

            image_original = spec.original
            draw = ImageDraw.Draw(image_original)

            # calculate detection boxes in original image coordinates
            dataframes = det.pandas()
            dataframe = dataframes.xyxy[0]
            predictions = dataframe.itertuples(index=False)

            for prediction in predictions:
                w_original, h_original = spec.original.size

                scale_x = w_original / (self.flower_model_dim[0] - spec.horizontal_pad)
                scale_y = h_original / (self.flower_model_dim[1] - spec.vertical_pad)

                x_max = prediction.xmax - spec.horizontal_pad / 2
                y_max = prediction.ymax - spec.vertical_pad   / 2
                x_min = prediction.xmin - spec.horizontal_pad / 2
                y_min = prediction.ymin - spec.vertical_pad   / 2

                box = DetectionBox(x_max, y_max, x_min, y_min)
                box = box.scale(scale_x, scale_y) \
                        .add_margin(10, w_original, h_original)

                cropped_image = spec.original.crop(box.to_tuple())
                cropped_images.append(cropped_image)

        return cropped_images

                # draw.rectangle(box.to_tuple(), width=8, outline='blue')
                # image_original.show()


    def predict_flowers(self, dataloader: DataLoader) -> list[Image.Image]:
        all_cropped_images: list[Image.Image] = []

        with torch.no_grad():
            for image_specs in dataloader:
                resized_images = [spec.resized for spec in image_specs]
                detections = self._run_inference(resized_images)

                cropped_images = self._postprocessing(detections, image_specs)
                all_cropped_images.extend(cropped_images)

        return all_cropped_images


    @TimeMeasure(name="pollinator_inference")
    def _pollinator_inference(self, images):
        return self.pollinator_model(images)

    
    def predict_pollinators(self, dataloader: DataLoader):
        with torch.no_grad():
            for images_specs in dataloader:
                resized_images = [spec.resized for spec in images_specs]
                detections = self._pollinator_inference(resized_images)
                #for (det, _) in zip(detections.tolist(), images_specs):

                    # calculate detection boxes in original image coordinates
                    #for row in det.pandas().xyxy[0].itertuples(index=False):
                        #if row.confidence > 0.8:
                            #detections.show()

