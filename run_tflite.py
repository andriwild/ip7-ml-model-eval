import os
import time
from helpers.csv_writer import CSVWriter
from multiprocessing import cpu_count
from PIL import ImageDraw, ImageFont

import numpy as np
import tensorflow as tf
from PIL import Image

# Constants
IMAGE_DIR = "/home/andri/repos/ip7-ml-model-eval/images/root/"
MODEL1_PATH ="mitwelten_models/flowers_ds_v5_640_yolov5n_v0_cnv-fp16.tflite"
MODEL2_PATH ="mitwelten_models/pollinators_ds_v6_480_yolov5s_bs32_300ep_multiscale_v0-fp16.tflite"
CONFIDENCE_THRESHOLD = 0.5
DISPLAY_DETECTIONS = True # Set this flag to True to display detections
NUM_THREADS = cpu_count() # Set the number of threads for interpreter if needed
image_show_counter = 0

# Load the models
def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    return interpreter

interpreter1 = load_interpreter(MODEL1_PATH)
interpreter2 = load_interpreter(MODEL2_PATH)

# Get input and output details for model
def get_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

input_details1, output_details1 = get_model_details(interpreter1)
input_details2, output_details2 = get_model_details(interpreter2)

# Function to preprocess image for the model
def preprocess_image(image, input_details):
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    image = image.resize((width, height))
    input_data = np.expand_dims(image, axis=0)
    if input_details[0]['dtype'] == np.float32:
        input_data = np.array(input_data, dtype=np.float32) / 255.0
    elif input_details[0]['dtype'] == np.uint8:
        input_data = np.array(input_data, dtype=np.uint8)
    return input_data

# Function to perform inference on an interpreter
def run_inference(interpreter, input_data):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output_data

# Function to process detections from model 1
def process_detections_model1(output_data):
    output_data = np.squeeze(output_data)  # Shape: [N, 8]

    boxes = []
    scores = []
    classes = []

    for detection in output_data:
        confidence = detection[4]
        if confidence > CONFIDENCE_THRESHOLD:
            # Extract bounding box coordinates
            x_center = detection[0]
            y_center = detection[1]
            width = detection[2]
            height = detection[3]

            # Convert to xmin, ymin, xmax, ymax
            xmin = x_center - (width / 2)
            xmax = x_center + (width / 2)
            ymin = y_center - (height / 2)
            ymax = y_center + (height / 2)

            # Ensure coordinates are within [0,1]
            xmin = max(0, min(1, xmin))
            xmax = max(0, min(1, xmax))
            ymin = max(0, min(1, ymin))
            ymax = max(0, min(1, ymax))

            # Get class probabilities
            class_probs = detection[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]

            boxes.append([ymin, xmin, ymax, xmax])
            scores.append(confidence * class_score)
            classes.append(class_id)

    return boxes, scores, classes

# Function to process detections from model 2
def process_detections_model2(output_data):
    detections = []
    output_data = np.squeeze(output_data)  # Adjust based on the output shape

    for detection in output_data:
        confidence = detection[4]
        if confidence > CONFIDENCE_THRESHOLD:
            # Extract bounding box coordinates
            x_center = detection[0]
            y_center = detection[1]
            width = detection[2]
            height = detection[3]

            # Convert to xmin, ymin, xmax, ymax
            xmin = x_center - (width / 2)
            xmax = x_center + (width / 2)
            ymin = y_center - (height / 2)
            ymax = y_center + (height / 2)

            # Ensure coordinates are within [0,1]
            xmin = max(0, min(1, xmin))
            xmax = max(0, min(1, xmax))
            ymin = max(0, min(1, ymin))
            ymax = max(0, min(1, ymax))

            # Get class probabilities
            class_probs = detection[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]

            detections.append({
                'box': [ymin, xmin, ymax, xmax],
                'score': confidence * class_score,
                'class': class_id
            })
    return detections

# Directory containing images
image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.lower().endswith(('.jpg', '.png'))]

# Initialize CSV writer for measurements
csv_writer = CSVWriter('data/test/measurements.csv', ['Image', 'Model1 Time', 'Model2 Time', 'Total Time'])

for image_path in image_paths:
    start_total = time.time()
    image_name = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB')

    # Preprocess image for model 1
    input_data1 = preprocess_image(image, input_details1)

    # Run inference for model 1
    start_model1 = time.time()
    output_data1 = run_inference(interpreter1, input_data1)
    end_model1 = time.time()

    # Process detections from model 1
    boxes, scores, classes = process_detections_model1(output_data1)

    # Process detected boxes
    cropped_images = []
    image_width, image_height = image.size
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[i]

        # Convert normalized coordinates to pixel values
        left = int(xmin * image_width)
        top = int(ymin * image_height)
        right = int(xmax * image_width)
        bottom = int(ymax * image_height)

        if right <= left or bottom <= top:
            continue  # Skip invalid boxes

        cropped_image = image.crop((left, top, right, bottom))
        cropped_images.append((cropped_image, i))  # Keep index for reference

    # Run model 2 on cropped images
    model2_time = 0
    if cropped_images:
        batch_size_dim = input_details2[0]['shape'][0]
        if batch_size_dim == -1 or batch_size_dim is None:
            # Model supports variable batch size
            batch_input_data = []
            for cropped_image, _ in cropped_images:
                input_data2 = preprocess_image(cropped_image, input_details2)
                batch_input_data.append(input_data2[0])
            batch_input_data = np.stack(batch_input_data)

            interpreter2.resize_tensor_input(input_details2[0]['index'], [len(cropped_images), input_details2[0]['shape'][1], input_details2[0]['shape'][2], input_details2[0]['shape'][3]])
            interpreter2.allocate_tensors()

            # Run inference for model 2
            start_model2 = time.time()
            interpreter2.set_tensor(input_details2[0]['index'], batch_input_data)
            interpreter2.invoke()
            end_model2 = time.time()

            model2_time = end_model2 - start_model2

            output_data2 = interpreter2.get_tensor(output_details2[0]['index'])
            # Process output_data2 as needed
            for idx, (cropped_image, orig_index) in enumerate(cropped_images):
                detections = process_detections_model2(output_data2[idx])
                # If DISPLAY_DETECTIONS is True and detections are found, display the cropped image
                if DISPLAY_DETECTIONS and detections and image_show_counter < 10:

                    draw = ImageDraw.Draw(cropped_image)
                    for det in detections:
                        ymin, xmin, ymax, xmax = det['box']
                        # Convert normalized coordinates to pixel values
                        left = int(xmin * cropped_image.width)
                        top = int(ymin * cropped_image.height)
                        right = int(xmax * cropped_image.width)
                        bottom = int(ymax * cropped_image.height)
                   
                        # Draw the bounding box
                        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
                   
                        # Draw the label
                        label = f"Class {det['class']}: {det['score']:.2f}"
                        draw.text((left, top - 10), label, fill="red")
                    
                    cropped_image.show()
                    image_show_counter += 1
        else:
            # Model does not support variable batch size
            for idx, (cropped_image, orig_index) in enumerate(cropped_images):
                input_data2 = preprocess_image(cropped_image, input_details2)

                # Run inference for model 2
                start_model2 = time.time()
                output_data2 = run_inference(interpreter2, input_data2)
                end_model2 = time.time()

                model2_time += (end_model2 - start_model2)

                detections = process_detections_model2(output_data2)
                # If DISPLAY_DETECTIONS is True and detections are found, display the cropped image
                if DISPLAY_DETECTIONS and detections and image_show_counter < 10:

                    draw = ImageDraw.Draw(cropped_image)
                    for det in detections:
                        ymin, xmin, ymax, xmax = det['box']
                        # Convert normalized coordinates to pixel values
                        left = int(xmin * cropped_image.width)
                        top = int(ymin * cropped_image.height)
                        right = int(xmax * cropped_image.width)
                        bottom = int(ymax * cropped_image.height)
                   
                        # Draw the bounding box
                        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
                   
                        # Draw the label
                        label = f"Class {det['class']}: {det['score']:.2f}"
                        draw.text((left, top - 10), label, fill="red")
                    
                    cropped_image.show()
                    image_show_counter += 1
    else:
        model2_time = 0

    end_total = time.time()
    total_time = end_total - start_total
    model1_time = end_model1 - start_model1

    # Append measurements to CSV
    csv_writer.append_data([image_name, model1_time, model2_time, total_time])

# Flush CSV data
csv_writer.flush()

