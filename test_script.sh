#!/bin/bash

python run_cpu.py --threads 1 --dataset_size 20 --batch_size 1
python run_cpu.py --threads 2 --dataset_size 20 --batch_size 1
python run_cpu.py --threads 4 --dataset_size 20 --batch_size 1
python run_cpu.py --threads 8 --dataset_size 20 --batch_size 1

python run_cpu.py --threads 1 --dataset_size 20 --batch_size 2
python run_cpu.py --threads 2 --dataset_size 20 --batch_size 2
python run_cpu.py --threads 4 --dataset_size 20 --batch_size 2
python run_cpu.py --threads 8 --dataset_size 20 --batch_size 2

python run_cpu.py --threads 1 --dataset_size 20 --batch_size 4
python run_cpu.py --threads 2 --dataset_size 20 --batch_size 4
python run_cpu.py --threads 4 --dataset_size 20 --batch_size 4
python run_cpu.py --threads 8 --dataset_size 20 --batch_size 4

python run_cpu.py --threads 1 --dataset_size 20 --batch_size 8
python run_cpu.py --threads 2 --dataset_size 20 --batch_size 8
python run_cpu.py --threads 4 --dataset_size 20 --batch_size 8
python run_cpu.py --threads 8 --dataset_size 20 --batch_size 8
