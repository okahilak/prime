#!/bin/bash

python3 -m venv .venv && source .venv/bin/activate

read -rp "Run preprocessing? [y/N] " run_preprocessing
if [[ "$run_preprocessing" =~ ^[Yy]$ ]]; then
    time python3 online_preprocessing/run_all_subjects.py | tee preprocessing.log
fi
time python3 train_transfer.py -c configs/replicate_prime.yaml | tee training.log
time python3 train_transfer.py -c configs/replicate_prime_short.yaml | tee training.log
