import os
import argparse
import cv2
import numpy as np
import sys
from typing import List
from tflite_runtime.interpreter import Interpreter

def load_labels(labelmap_path: str) -> List[str]:
    """Loads labels from a label map file."""
    try:
        with open(labelmap_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        if labels[0] == '???':
            labels.pop(0)
        return labels
    except IOError as e:
        print(f"Error reading label map file: {e}")
        sys.exit()

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', required=True, help='Folder where the .tflite file is located')
    parser.add_argument('--graph', default='detect.tflite', help='Name of the .tflite file')
    parser.add_argument('--labels', default='labelmap.txt', help='Name of the label map file')
    parser.add_argument('--threshold', default='0.5', help='Minimum confidence threshold')
    parser.add_argument('--image', required=True, help='Path to the input image')
    args = parser.parse_args()

    # Configuration
    model_path = os.path.join(os.getcwd(), args.modeldir, args.graph)
    labelmap_path = os.path.join(os.getcwd(), args.modeldir, args.labels)
    min_conf_threshold = float(args.threshold)

    # Load labels and interpreter
    labels = load_labels(labelmap_path)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1:3]
    floating_model = (input_details[0]['dtype'] == np.float32)

    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:  # For newer models
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Read image
    image_path = args.image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read input image {image_path}")
        sys.exit()
    resH, resW, _ = frame.shape

    # Prepare input image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence scores

    # Loop over detections and draw boxes
    detections = []
    for i in range(len(scores)):
        if min_conf_threshold < scores[i] <= 1.0:
            # Calculate bounding box coordinates
            ymin = int(max(1, (boxes[i][0] * resH)))
            xmin = int(max(1, (boxes[i][1] * resW)))
            ymax = int(min(resH, (boxes[i][2] * resH)))
            xmax = int(min(resW, (boxes[i][3] * resW)))

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Prepare label
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            detections.append(label)

            # Draw label background and text
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Output detections
    if detections:
        print("Detections:")
        for det in detections:
            print(det)
    else:
        print("No objects detected.")

    # Save the image with detections
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, frame)
    print(f"Output image saved to {output_image_path}")

if __name__ == "__main__":
    main()
