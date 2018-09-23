#!/bin/bash
echo -e '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
for d in models/* ; do
    if [ -d $d ] ; then
        echo "====================== $d ======================"
        if [ -f reports/$(basename $d).txt ] ; then
            grep --with-filename ' score:' reports/$(basename $d).txt || \
                grep --with-filename 'Accuracy:' reports/$(basename $d).txt || \
                grep --with-filename 'R^2:' reports/$(basename $d).txt
        else
            grep --with-filename ' score:' $d/train.*.log || \
                grep --with-filename 'Accuracy:' $d/train.*.log || \
                grep --with-filename 'R^2:' $d/train.*.log
        fi
    fi
done
