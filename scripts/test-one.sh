#!/bin/bash

if (( $# != 1 )); then
    echo "error: what image do you want to test for?"
    exit 1
fi

# determine random preprocessing parameters
rot="$( echo "scale=2 ; $RANDOM / 32767 * 20 - 10" | bc )"
scl="$( echo "scale=2 ; $RANDOM / 32767 * 50 + 250" | bc )"

# convert the image to pgm and do some random preprocessing
convert "$1" -background black -scale "$scl" -rotate "$rot" ./tmp.pgm

# test it
./main test ./tmp.pgm 10

# remove image
rm tmp.pgm
