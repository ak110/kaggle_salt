#!/bin/bash
set -eux
# 作業用シェルスクリプト。
cd solution
./run.sh stack
python3 stack.py predict
