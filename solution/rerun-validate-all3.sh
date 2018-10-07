#!/bin/bash
set -eux
python3 darknet53_sepscse.py validate --tta
python3 bin_nas.py validate --tta
python3 reg_nas.py validate --tta

python3 darknet53_sepscse.py predict --tta
python3 bin_nas.py predict --tta
python3 reg_nas.py predict --tta
