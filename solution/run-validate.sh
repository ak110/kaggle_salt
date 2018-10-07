#!/bin/bash
set -eux
PYFILE=${1}.py
if [ ! -e $PYFILE ] ; then echo "Error: $PYFILE is not found!" 2>&1 ; exit 1 ; fi

python3 $PYFILE validate --tta
python3 $PYFILE predict --tta
