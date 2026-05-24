#!/bin/bash

python3 -m venv .venv && source .venv/bin/activate
time python3 online_preprocessing/run_all_subjects.py | tee preprocessing.log
time python3 train_transfer.py -c configs/replicate_prime.yaml | tee training.log
