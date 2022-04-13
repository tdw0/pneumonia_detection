#!/bin/bash

if [[ -e ZhangLabData.zip ]]; then
    echo "ZhangLabData.zip already exists, skipping download..."
else
    curl -L https://data.mendeley.com/archiver/rscbjbr9sj?version=3 | zcat | tee > ZhangLabData.zip
fi

sha256sum ZhangLabData.zip
echo "acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563"
echo "Checksums must match!!"
