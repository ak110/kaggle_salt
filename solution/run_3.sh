#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)

mpirun -np $GPUS python3 model_3.py --cv-index=0 $*
mpirun -np $GPUS python3 model_3.py --cv-index=1 $*
mpirun -np $GPUS python3 model_3.py --cv-index=2 $*
mpirun -np $GPUS python3 model_3.py --cv-index=3 $*
mpirun -np $GPUS python3 model_3.py --cv-index=4 $*
python3 report.py model_3
