#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)

mpirun -np $GPUS python3 train.py
python3 predict.py

