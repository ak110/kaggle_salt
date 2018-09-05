#!/bin/bash
set -eux
# 作業用シェルスクリプト。
cd solution
python3 bin.py validate
python3 fast.py validate
python3 darknet53_nr.py validate
python3 darknet53.py validate
python3 resnet.py validate
python3 yolo.py validate
python3 vgg.py validate
python3 nasnet.py validate
