#!/bin/bash
set -eux
python3 darknet53_coord_hcs.py validate
python3 darknet53_mixup.py validate
