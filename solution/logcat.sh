#!/bin/bash
echo -e '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
for d in models/* ; do
    if [ -d $d ] ; then
        echo "====================== $(basename $d).py ======================"
        if [ -f models/$(basename $d)/validate.log -a "$1" != "--all" ] ; then
            grep --with-filename ' score:' models/$(basename $d)/validate.log || \
                grep --with-filename 'Accuracy:' models/$(basename $d)/validate.log || \
                grep --with-filename 'R^2:' models/$(basename $d)/validate.log
        else
            grep --with-filename ' score:' $d/train.*.log || \
                grep --with-filename 'Accuracy:' $d/train.*.log || \
                grep --with-filename 'R^2:' $d/train.*.log
            grep --with-filename ' score:' $d/fine.*.log || \
                grep --with-filename 'Accuracy:' $d/fine.*.log || \
                grep --with-filename 'R^2:' $d/fine.*.log
        fi
    fi
done
