#!/bin/bash
echo -e '\n\n\n'
for d in cache/val/*.pkl cache/test/*.pkl ; do
    if [ ! -f $(basename $d .pkl).py ] ; then
        echo "rm -f $d"
    fi
done
for d in models/* ; do
    if [ ! -f $(basename $d).py ] ; then
        echo "rm -rf $d"
    fi
done
for d in reports/*.txt ; do
    if [ ! -f "$(basename $d .txt).py" ] ; then
        echo "rm -f $d"
    fi
done
echo -e '\n\n\n'
