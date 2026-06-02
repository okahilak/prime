#!/bin/bash

read -rp "Run preprocessing? [y/N] " run_preprocessing
if [[ "$run_preprocessing" =~ ^[Yy]$ ]]; then
    python3 online_preprocessing/run_all_subjects.py
fi
python3 -m prime.train --cv

# Training datasets from the first fold of the cross-validation split
python3 -m prime.train --train 18 19 20 23 25 28 29 35 40 42 43 44 45 50 101 102 107 108 110 111 113 115 116 120 125

# Test datasets from the first fold of the cross-validation split
python3 -m prime.train --test 21 22 24 26 27 31 34 36 38

# Loop over test datasets
for subject_id in 21 22 24 26 27 31 34 36 38; do
    python3 -m prime.simulate_online "$subject_id"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "ERROR: simulate_online.py failed for subject ${subject_id} (exit code ${exit_code}). Aborting."
        exit 1
    fi
done
echo "All simulate_online.py checks passed."
