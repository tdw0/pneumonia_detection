#!/bin/bash

if (( $# != 1 )); then
    echo "error: how many images do you want to test for?"
    echo "usage: $0 10"
    echo "       $0 100"
    exit 1
fi

for type in NORMAL "BACTERIA|VIRUS" ; do
    echo "-------------------"
    echo "type: $type"
    find prepped_dataset/test -type f -print0 | egrep -z -i "pgm$" | egrep -z -i "$type" | \
        sort -z -R | head -z -n "$1" | \
            xargs -0 ./main predict
    echo "-------------------"
done
