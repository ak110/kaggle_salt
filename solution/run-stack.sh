#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)
PYFILE=stack.py
mpirun -np $GPUS python3 $PYFILE train --cv-index=0
mpirun -np $GPUS python3 $PYFILE train --cv-index=1
mpirun -np $GPUS python3 $PYFILE train --cv-index=2
mpirun -np $GPUS python3 $PYFILE train --cv-index=3
mpirun -np $GPUS python3 $PYFILE train --cv-index=4
mpirun -np $GPUS python3 $PYFILE train --cv-index=5
mpirun -np $GPUS python3 $PYFILE train --cv-index=6
mpirun -np $GPUS python3 $PYFILE train --cv-index=7
mpirun -np $GPUS python3 $PYFILE train --cv-index=8
mpirun -np $GPUS python3 $PYFILE train --cv-index=9
mpirun -np $GPUS python3 $PYFILE train --cv-index=10
mpirun -np $GPUS python3 $PYFILE train --cv-index=11
mpirun -np $GPUS python3 $PYFILE train --cv-index=12
mpirun -np $GPUS python3 $PYFILE train --cv-index=13
mpirun -np $GPUS python3 $PYFILE train --cv-index=14
python3 $PYFILE validate
python3 $PYFILE predict
