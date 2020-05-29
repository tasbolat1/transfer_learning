#!/bin/sh

# activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
# conda activate my_env
conda activate

# run python script with different cut_idx
for i in 10, 20, 50, 100, 200, 300, 400
do
    python BioTac_framework.py --cutidx $i
done