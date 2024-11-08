import os
import argparse
import numpy as np
import sys
import time
import csv
from typing import List
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw
from termcolor import cprint  # Falls für farbige Ausgaben benötigt

def load_all_images(folder_path: str, num_images: int) -> List[str]:
    images = []
    img_ext = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in img_ext):
            img_path = os.path.join(folder_path, file)
            images.append(img_path)
            if len(images) >= num_images:
                break

    return images

def write_results(results, output_file):
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results)
    cprint(f"Results written to {output_file}", "green")

def load_labels(label_path: str) -> List[str]:
    """Lädt Labels aus einer Datei."""
    try:
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        if labels[0] == '???':
            labels.pop(0)
        return labels
    except IOError as e:
        print(f"Fehler beim Lesen der Label-Datei: {e}")
        sys.exit()

def main():
    # Argumente parsen
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Pfad zur .tflite-Modelldatei')
    parser.add_argument('--labels', required=True, help='Pfad zur Label-Datei')
    parser.add_argument('--threshold', default='0.5', help='Minimale Konfidenzschwelle')
    parser.add_argument('--image_dir', required=True, help='Verzeichnis mit den Bildern')
    parser.add_argument('--num_images', type=int, default=10, help='Anzahl der zu verarbeitenden Bilder')
    parser.add_argument('--output_dir', default='output', help='Verzeichnis zum Speichern der annotierten Bilder')
    parser.add_argument('--results_file', default='results.csv', help='CSV-Datei zum Speichern der Ergebnisse')
    args = parser.parse_args()

    # Konfiguration
    model_path = args.model
    label_path = args.labels
    min_conf_threshold = float(args.threshold)
    num_images = args.num_images
    output_dir = args.output_dir
    results_file = args.results_file

    # Labels und Interpreter laden
    labels = load_labels(label_path)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Modelldetails abrufen
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1:3]
    floating_model = (input_details[0]['dtype'] == np.float32)

    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:  # Für neuere Modelle
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Bilder laden
    image_paths = load_all_images(args.image_dir, num_images)
    if not image_paths:
        print(f"Keine Bilder im Verzeichnis {args.image_dir} gefunden.")
        sys.exit()

    # Output-Verzeichnis erstellen, falls nicht vorhanden
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Jedes Bild verarbeiten
    for image_path in image_paths:
        # Inferenzzeit starten
        start_time = time.time()

        # Bild lesen
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size

        # Bild auf Eingabegröße des Modells anpassen
        image_resized = image.resize((width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = np.uint8(input_data)

        # Inferenz ausführen
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Ergebnisse abrufen
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # Detections verarbeiten
        detections = []
        draw = ImageDraw.Draw(image)
        for i in range(len(scores)):
            if min_conf_threshold < scores[i] <= 1.0:
                # Bounding Box auf Originalbildgröße skalieren
                ymin = int(max(1, (boxes[i][0] * orig_height)))
                xmin = int(max(1, (boxes[i][1] * orig_width)))
                ymax = int(min(orig_height, (boxes[i][2] * orig_height)))
                xmax = int(min(orig_width, (boxes[i][3] * orig_width)))

                # Label vorbereiten
                object_name = labels[int(classes[i])]
                confidence = int(scores[i] * 100)
                label = f'{object_name}: {confidence}%'
                detections.append([image_path, object_name, confidence, xmin, ymin, xmax, ymax])

                # Bounding Box und Label zeichnen
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
                draw.text((xmin, ymin - 10), label, fill='red')

        # Inferenzzeit stoppen
        end_time = time.time()
        inference_time = end_time - start_time

        # Detections ausgeben
        if detections:
            print(f"Erkannte Objekte in {image_path}:")
            for det in detections:
                print(f" - {det[1]}: {det[2]}%")
        else:
            print(f"Keine Objekte in {image_path} erkannt.")

        # Annotiertes Bild speichern
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        image.save(output_image_path)
        print(f"Ausgabebild gespeichert unter {output_image_path}")

        # Ergebnisse in CSV-Datei schreiben
        for det in detections:
            # Inferenzzeit zu den Ergebnissen hinzufügen
            det.append(inference_time)
            write_results(det, results_file)

    print("Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    main()


