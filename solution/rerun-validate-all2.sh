#!/bin/bash
set -eux
python3 darknet53_large2.py validate --tta
python3 darknet53_resize128.py validate --tta
