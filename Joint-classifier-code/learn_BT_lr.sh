#!/bin/sh
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1 --num_epochs 1 --lr 0.0001 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.1 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.01 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.001 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.0001 --c1B 32 --c2B 64 --hB 18
