#!/bin/bash
set -eux
./run-validate.sh darknet53_large2
./run-validate.sh darknet53_resize128
