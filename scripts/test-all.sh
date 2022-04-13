#!/bin/bash

if (( $# != 1 )); then
    echo "error: unspecified batch size, how many number of images to test per run?"
    echo "usage: $0 10"
    echo "       $0 100"
    exit 1
fi

echo "total dataset size: $(find prepped_dataset/test -type f | wc -l)"

for type in NORMAL "BACTERIA|VIRUS" ; do
    echo "-------------------"
    echo "type: $type"
    find prepped_dataset/test -type f -print0 | egrep -z -i "pgm$" | egrep -z -i "$type" | \
        sort -z -R | \
            xargs -0 -L"$1" ./main predict
    echo "-------------------"
done
