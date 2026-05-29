#!/bin/bash

python3 -m venv .venv && source .venv/bin/activate

read -rp "Run preprocessing? [y/N] " run_preprocessing
if [[ "$run_preprocessing" =~ ^[Yy]$ ]]; then
    time python3 online_preprocessing/run_all_subjects.py | tee preprocessing.log
fi
time python3 train_transfer.py --cv -c configs/prime.yaml | tee training.log
