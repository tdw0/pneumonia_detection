#!/bin/bash
# This preps the dataset AND launches the program

if [[ -d CellData ]]; then
    echo "ZhangLabData.zip already extracted (skipping)"
else
    unzip ZhangLabData.zip
fi

# remove prepped dataset if exists
rm -rf ./prepped_dataset/ 2> /dev/null

mkdir prepped_dataset

# for each set in the dataset
for set in test train; do
    # create directory for the set
    mkdir "prepped_dataset/$set"

    # find all jpegs of the set and
    find "./CellData/chest_xray/$set" -type f -print0 | egrep -z -i "jpeg$" | sort -z -R | \
        while read -r -d $'\0' file; do
            basename="$( basename "$file" ".jpeg" )"

            # create multiple copies for the 
            for copy in {1..6}; do
                # generate random rotation angle/scale factor
                rot="$( echo "scale=2 ; $RANDOM / 32767 * 20 - 10" | bc )"
                scl="$( echo "scale=2 ; $RANDOM / 32767 * 50 + 250" | bc )"

                new_filename="$basename-$copy.pgm"

                # copy file
                printf "convert \"$file\" -background black -rotate \"$rot\" -scale \"$scl\" \"./prepped_dataset/$set/$new_filename\" && printf \".\"\x00"
            done
        done | time xargs -0 -L1 -P8 sh -c
done

echo -e "\ndone!"
