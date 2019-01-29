#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)
PYFILE=${1}.py
python3 $PYFILE validate
python3 $PYFILE predict
