import logging

from csv_logger import CsvLogger

logger = CsvLogger(
    filename=f'reports/inference-test.csv',
    delimiter=',',
    level=logging.INFO,
    add_level_nums=None,
    fmt=f'%(asctime)s,%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    max_size=1024,
    max_files=4,
    header=['date', 'device', 'model', 'time_ms', 'dataset_size', 'dataset']
)