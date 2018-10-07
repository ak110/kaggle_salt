#!/bin/bash
set -eux
PYFILE=${1}.py
if [ ! -e $PYFILE ] ; then echo "Error: $PYFILE is not found!" 2>&1 ; exit 1 ; fi

GPUS=$(nvidia-smi --list-gpus | wc -l)

mpirun -np $GPUS python3 $PYFILE train --cv-index=0
mpirun -np $GPUS python3 $PYFILE train --cv-index=1
mpirun -np $GPUS python3 $PYFILE train --cv-index=2
mpirun -np $GPUS python3 $PYFILE train --cv-index=3
mpirun -np $GPUS python3 $PYFILE train --cv-index=4

python3 $PYFILE validate
python3 $PYFILE predict
