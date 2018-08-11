#!/bin/bash
set -eux
if [ $# -eq 0 ] ; then echo 'Error: run python file!' 2>&1 ; exit 1 ; fi
GPUS=$(nvidia-smi --list-gpus | wc -l)
mpirun -np $GPUS python3 $* train --cv-index=0
mpirun -np $GPUS python3 $* train --cv-index=1
mpirun -np $GPUS python3 $* train --cv-index=2
mpirun -np $GPUS python3 $* train --cv-index=3
mpirun -np $GPUS python3 $* train --cv-index=4
python3 $* validate
python3 $* predict
