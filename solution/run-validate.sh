#!/bin/bash
set -eux
PYFILE=${1}.py
if [ ! -e $PYFILE ] ; then echo "Error: $PYFILE is not found!" 2>&1 ; exit 1 ; fi

rm -rfv reports/${1}.txt cache/*/${1}.pkl || true

python3 $PYFILE validate
python3 $PYFILE predict
