import csv
import os
from pathlib import Path

class CSVWriter:
    def __init__(self, filename, header):
        self.filename = filename
        self.header = header
        print(filename.rsplit("/", 1)[0])

        Path(filename.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)

        # Check if file exists; if not, write header
        file_exists = os.path.isfile(self.filename)
        if not file_exists:
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.header)

    def append_data(self, data):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

if __name__ == "__main__":
    header = ['Function', 'Execution Time (seconds)']
    csv_writer = CSVWriter('measurement.csv', header)

    csv_writer.append_data(['function1', 0.123])
    csv_writer.append_data(['function2', 0.456])
