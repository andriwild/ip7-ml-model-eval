import logging
import platform
from datetime import datetime

import time
from time import sleep


def Measure(func):
    def inner1(*args, **kwargs):
        start_time: float = time.perf_counter()
        func(*args, **kwargs)
        end_time: float = time.perf_counter()
        return end_time - start_time
    return inner1

# @Measure
# def main():
#     sleep(2)
#
#
# if __name__ == "__main__":
#     r = main()
#     print(r)
