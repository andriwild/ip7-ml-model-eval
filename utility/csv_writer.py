import csv
import os
from pathlib import Path
from typing import List

class CSVWriter:
    def __init__(self, filename, header, cache_mode=True):
        self.filename = filename
        self.header = header
        self.cache_mode = cache_mode
        self.cache_data = []

        Path(filename.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)

        # Check if file exists; if not, write header
        file_exists = os.path.isfile(self.filename)
        if not file_exists:
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.header)

    def append_data(self, data: List):
        if self.cache_mode:
            self.cache_data.append(data)
        else:
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)

    def flush(self):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for data in self.cache_data:
                writer.writerow(data)
        self.cache_data = []

