#!/bin/sh

python learn.py --sensor BioTac --lr 0.0001 --num_epochs 2000 --cut_idx 400

python learn.py --sensor Icub --lr 0.001 --num_epochs 2000
