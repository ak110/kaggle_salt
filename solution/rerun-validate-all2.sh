#!/bin/bash
set -eux
python3 darknet53_large2.py validate
python3 darknet53_resize128.py validate
