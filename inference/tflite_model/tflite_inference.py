# some parts of this code are taken from:
# https://github.com/karanjakhar/yolov5-export-to-raspberry-pi
from os import cpu_count
import tensorflow as tf
import numpy as np
from PIL import Image
import time

class YoloModel:
    def __init__(
        self,
        weights="models/flower.tflite",
        image_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=["class1", "class2", "class3"],
        margin=10,
    ):

        self.weights = weights
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.interpreter = tf.lite.Interpreter(self.weights, num_threads=cpu_count())
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.max_detections=25 # tflite
        self.names = classes
        self.margin = margin
        self.total_inference_time = 0
        self.number_of_inferences = 0

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, boxes, scores, threshold):
        assert boxes.shape[0] == scores.shape[0]
        # bottom-left origin
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]
        # top-right target
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]
        # box coordinate ranges are inclusive-inclusive
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []
        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = self.compute_iou(
                boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes]
            )
            filtered_indexes = set((ious > threshold).nonzero()[0])
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes) if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)

    def compute_iou(self, box, boxes, box_area, boxes_area):
        # this is the iou of the box against all other boxes
        assert boxes.shape[0] == boxes_area.shape[0]
        # get all the origin-ys
        # push up all the lower origin-xs, while keeping the higher origin-xs
        ys1 = np.maximum(box[0], boxes[:, 0])
        # get all the origin-xs
        # push right all the lower origin-xs, while keeping higher origin-xs
        xs1 = np.maximum(box[1], boxes[:, 1])
        # get all the target-ys
        # pull down all the higher target-ys, while keeping lower origin-ys
        ys2 = np.minimum(box[2], boxes[:, 2])
        # get all the target-xs
        # pull left all the higher target-xs, while keeping lower target-xs
        xs2 = np.minimum(box[3], boxes[:, 3])
        # each intersection area is calculated by the
        # pulled target-x minus the pushed origin-x
        # multiplying
        # pulled target-y minus the pushed origin-y
        # we ignore areas where the intersection side would be negative
        # this is done by using maxing the side length by 0
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
        # each union is then the box area
        # added to each other box area minusing their intersection calculated above
        unions = box_area + boxes_area - intersections
        # element wise division
        # if the intersection is 0, then their ratio is 0
        ious = intersections / unions
        return ious

    def nms(self, prediction):

        prediction = prediction[prediction[..., 4] > self.conf_thres]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = self.xywh2xyxy(prediction[:, :4])

        res = self.non_max_suppression(boxes, prediction[:, 4], self.iou_thres)

        result_boxes = []
        result_scores = []
        result_class_names = []
        for r in res:
            result_boxes.append(boxes[r])
            result_scores.append(prediction[r, 4])
            result_class_names.append(self.names[np.argmax(prediction[r, 5:])])

        return result_boxes, result_scores, result_class_names

    def _predict(self, image):
        t0 = time.time()

        original_size = image.shape[:2]
        input_data = np.ndarray(
            shape=(1, self.image_size, self.image_size, 3), dtype=np.float32
        )
        # image = cv2.resize(image,(self.image_size,self.image_size))
        # input_data[0] = image.astype(np.float32)/255.0
        input_data[0] = image

        # self.interpreter.allocate_tensors()

        # Get input and output tensors
        # input_details = self.interpreter.get_input_details()
        # output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Denormalize xywh
        pred[..., 0] *= original_size[1]  # x
        pred[..., 1] *= original_size[0]  # y
        pred[..., 2] *= original_size[1]  # w
        pred[..., 3] *= original_size[0]  # h

        result_boxes, result_scores, result_class_names = self.nms(pred)
        self.total_inference_time += time.time() - t0
        self.number_of_inferences += 1
        return result_boxes, result_scores, result_class_names

    def detect(self, image):
        if type(image) == str:
            image = Image.open(image)
        image_size_orig = image.size
        model_size = (self.image_size, self.image_size)
        image_lb = letterbox_image(image, model_size)
        image_array = np.asarray(image_lb)
        image_array_norm = image_array.astype(np.float32) / 255.0
        result_boxes, result_scores, result_class_names = self._predict(
            image_array_norm
        )
        if len(result_boxes) > 0:
            result_boxes = scale_coords(
                model_size,
                np.array(result_boxes),
                (image_size_orig[1], image_size_orig[0]),
            )

        return result_boxes, result_scores, result_class_names

        

    def get_crops(self, image):
        if type(image) == str:
            image = Image.open(image)
        result_boxes, result_scores, result_class_names = self.detect(image)

        crops = []
        for i, r in enumerate(result_boxes):
            ymin = int(r[1]) - self.margin
            if ymin < 0:
                ymin = 0
            ymax = int(r[3]) + self.margin
            if ymax > image.size[1]:
                ymax = image.size[1]
            xmin = int(r[0]) - self.margin
            if xmin < 0:
                xmin = 0
            xmax = int(r[2]) + self.margin
            if xmax > image.size[0]:
                xmax = image.size[0]
            crop = image.crop((xmin, ymin, xmax, ymax))
            crops.append(crop)

        return crops, result_class_names, result_scores



def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.Resampling.BOX)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
