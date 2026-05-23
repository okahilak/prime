#!/bin/bash

time python3 online_preprocessing/run_all_subjects.py
time python3 train_transfer.py -c configs/replicate_prime.yaml | tee replicate_prime.log
