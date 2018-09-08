#!/bin/bash
echo -e '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
for d in models/* ; do
    if [ -d $d ] ; then
        echo "====================== $d ======================"
        grep 'max score:' $d/train.*.log || grep --with-filename 'Accuracy:' $d/train.*.log
        if [ -f reports/$(basename $d).txt ] ; then grep --with-filename 'max score:' reports/$(basename $d).txt ; fi
    fi
done
