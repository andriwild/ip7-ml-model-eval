import logging
import platform
from datetime import datetime
import csv

import time
from time import sleep


def Measure2(func):
    def inner1(*args, **kwargs):
        start_time: float = time.perf_counter()
        func(*args, **kwargs)
        end_time: float = time.perf_counter()
        return end_time - start_time
    return inner1

def TimeMeasure(name=None, active=True):
    def decorator(func):
        func_name = name if name else func.__name__

        def wrapper(*args, **kwargs):
            if not active:
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            duration = end - start
            print(*args)
            if func_name == 'flower_inference':
                print("flower_inference")
                for r in result.tolist():
                    print(len(r.pandas().xyxy[0]))

            with open(f'measurement/ml-pipline.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([func_name, duration])

            return result
        return wrapper
    return decorator

