#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)
PYFILE=${1}.py
mpirun -np $GPUS python3 $PYFILE train --cv-index=0
python3 $PYFILE validate
python3 $PYFILE predict
