#!/bin/sh

# # activate conda environment
# source ~/anaconda3/etc/profile.d/conda.sh
# # conda activate my_env
# conda activate

# run python script with different cut_idx
for i in true false
do
    python BioTac_framework.py --cutidx 400 --lstm $i
    # if $i
    # then
    #     echo "i is true"
    # else
    #     echo "i is false"
    # fi



done