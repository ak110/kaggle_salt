#!/bin/bash
set -eux
./run-validate.sh darknet53_sepscse
./run-validate.sh bin_nas
./run-validate.sh reg_nas
