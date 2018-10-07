#!/bin/bash
set -eux
./run-validate.sh darknet53_coord_hcs
./run-validate.sh darknet53_mixup
./run-validate.sh darknet53_sepscse
