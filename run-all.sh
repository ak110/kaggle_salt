#!/bin/bash
set -eux
# 作業用シェルスクリプト。
cd solution
./run.sh bin
./run.sh fast
./run.sh darknet53_nr
./run.sh darknet53
./run.sh resnet
./run.sh yolo
./run.sh vgg
./run.sh nasnet
