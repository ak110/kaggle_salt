#!/bin/bash
set -eux
python3 bin_nas.py validate
python3 reg_nas.py validate
