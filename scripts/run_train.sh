#!/bin/bash

python src/train.py --task sst2 --mode full
python src/train.py --task sst2 --mode lora --rank 8
python src/evaluate.py --task sst2 --rank_sweep