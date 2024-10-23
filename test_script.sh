#!/bin/bash

# python run_cpu.py --threads 1 --dataset_size 5 --batch_size 1
# python run_cpu.py --threads 2 --dataset_size 5 --batch_size 1
# python run_cpu.py --threads 4 --dataset_size 5 --batch_size 1
# python run_cpu.py --threads 8 --dataset_size 5 --batch_size 1

# python run_cpu.py --threads 1 --dataset_size 5 --batch_size 2
# python run_cpu.py --threads 2 --dataset_size 5 --batch_size 2
# python run_cpu.py --threads 4 --dataset_size 5 --batch_size 2
# python run_cpu.py --threads 8 --dataset_size 5 --batch_size 2

python run_cpu.py --threads 1 --dataset_size 10 --batch_size 4 --data_folder "$1"
python run_cpu.py --threads 2 --dataset_size 10 --batch_size 4 --data_folder "$1"
python run_cpu.py --threads 4 --dataset_size 10 --batch_size 4 --data_folder "$1"
python run_cpu.py --threads 8 --dataset_size 10 --batch_size 4 --data_folder "$1"

python run_cpu.py --threads 1 --dataset_size 10 --batch_size 8 --data_folder "$1"
python run_cpu.py --threads 2 --dataset_size 10 --batch_size 8 --data_folder "$1"
python run_cpu.py --threads 4 --dataset_size 10 --batch_size 8 --data_folder "$1"
python run_cpu.py --threads 8 --dataset_size 10 --batch_size 8 --data_folder "$1"
