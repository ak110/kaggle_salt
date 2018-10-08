#!/bin/bash
set -eux
PYFILE=${1}.py
python3 $PYFILE validate
python3 $PYFILE predict
