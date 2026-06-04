#!/bin/bash

read -rp "Run preprocessing? [y/N] " run_preprocessing
if [[ "$run_preprocessing" =~ ^[Yy]$ ]]; then
    python3 -m prime_core.preprocessing.run_all_subjects
fi
python3 -m prime_core.train --cv

# Training datasets from the first fold of the cross-validation split
python3 -m prime_core.train --train 18 19 20 23 25 28 29 35 40 42 43 44 45 50 101 102 107 108 110 111 113 115 116 120 125

# Test datasets from the first fold of the cross-validation split

test_subjects="21 22 24 26 27 31 34 36 38"

python3 -m prime_core.train --test $test_subjects

# Loop over test datasets
for subject_id in $test_subjects; do
    python3 -u -m prime_core.test_by_trial "$subject_id" | tee "offline_results/test_by_trial_${subject_id}"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "ERROR: test_by_trial.py failed for subject ${subject_id} (exit code ${exit_code}). Aborting."
        exit 1
    fi
done
echo "All test_by_trial.py checks passed."
