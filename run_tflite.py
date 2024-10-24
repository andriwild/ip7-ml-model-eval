from inference.tflite_model.tflite_inference import YoloModel
from PIL import Image
import os
import gc
from tqdm import tqdm

def load_all_images(folder_path: str) -> list[str]:
    all_images = []
    img_ext: list[str] = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in img_ext):
            img_path: str = os.path.join(folder_path, file)
            all_images.append(img_path)

    return all_images


all_images = load_all_images("/home/andri/repos/ip7-ml-model-eval/images/root/")
#all_images = load_all_images("/home/andri/fhnw/MSE/IP7/ml/dataset/flower_kaggle/flower_dataset_v4_yolo/flower_dataset_v4_yolo/images/test/")

def capture_image(img_pointer):
    print("img_pointer", img_pointer)
    print(all_images[img_pointer])
    image = Image.open(all_images[img_pointer])
    return image

model_1 = YoloModel(
    "models/mitwelten_models/flowers_ds_v5_640_yolov5n_v0_cnv-fp16.tflite",
    640,
    0.5,
    0.5,
    classes=['daisy', 'wildemoere', 'flockenblume'],
    margin=20,
)
model_2 = YoloModel(
    "models/mitwelten_models/pollinators_ds_v6_480_yolov5s_bs32_300ep_multiscale_v0-fp16.tflite",
    480,
    0.8,
    0.5,
    classes=["honigbiene", "wildbiene","hummel","schwebfliege","fliege"],
    margin=20,
)


img_pointer = 0
total_number_of_images = 0
while True:

    image = capture_image(img_pointer)
    img_pointer += 1
    if img_pointer >= len(all_images):
        print("All images processed")
        exit(0)
    orig_width, orig_height = image.size
    crops, result_class_names, result_scores = model_1.get_crops(image)
    nr_flowers = len(result_class_names)
    print(f"Number of flowers: {nr_flowers}")
    pollinator_index = 0
    i = 0
    for i in tqdm(range(nr_flowers)):
        crop_width, crop_height = crops[i].size
        crops2, result_class_names2, result_scores2 = model_2.get_crops(crops[i])
        print(f"Number of pollinators: {len(result_class_names2)}")
        for j in range(len(result_class_names2)):
            image.show()
            input("Press Enter to continue...")

            crop2_width, crop2_height = crops2[j].size
            # index, flower_index, class_name, score, crop=None
            pollinator_index += 1

    gc.collect()
    total_number_of_images += 1


        # ymin, xmin, ymax, xmax = det['box']
        # # Convert normalized coordinates to pixel values
        # left = int(xmin * cropped_image.width)
        # top = int(ymin * cropped_image.height)
        # right = int(xmax * cropped_image.width)
        # bottom = int(ymax * cropped_image.height)

        # # Draw the bounding box
        # draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

        # # Draw the label
        # label = f"Class {det['class']}: {det['score']:.2f}"
        # draw.text((left, top - 10), label, fill="red")
