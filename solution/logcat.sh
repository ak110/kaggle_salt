#!/bin/bash
for d in models/* ; do
    if [ -d $d ] ; then
        echo "====================== $d ======================"
        grep 'max score:' $d/train.*.log || grep 'Accuracy:' $d/train.*.log
    fi
done

