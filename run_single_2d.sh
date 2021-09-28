#!/bin/bash

python ./lan2d.py  --jq1 20.0 20.0 --jq2 20.0 20.0 \
    --minima 0.0 1.0 -5.0 -10.0 --freqs 4.2 4.0 --delta 6.0 6.0 \
    --init-q -1.0 -1.0 --num-steps 10000 --temp 300 --mass 16.0 16.0 \
    --gamma 9.0 9.0 --output lan2D_1 --outfreq 100 --plot-surface