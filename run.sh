#!/bin/bash

python3 -m venv .venv && source .venv/bin/activate

read -rp "Run preprocessing? [y/N] " run_preprocessing
if [[ "$run_preprocessing" =~ ^[Yy]$ ]]; then
    python3 online_preprocessing/run_all_subjects.py
fi
python3 train.py --cv
