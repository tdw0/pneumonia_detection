#!/bin/bash

if (( $# != 1 )); then
    echo "error: How many images do you want per batch?"
    echo "usage: $0 100"
    echo "       $0 200"
    exit 1
fi

# train the network:
nice -n 20 ./main train "$1"
