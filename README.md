# Online PRIME

## Setting up online pipeline

In NeuroSimo, create a project named `prime` (creates directory `~/projects/prime`).

```bash
cd ~/projects/prime
rm -rf decider
git clone https://github.com/okahilak/prime.git decider

# Assuming Tübingen's samba share is mounted at ~/samba
rsync -ah --info=progress2 ~/samba/Projects/2026-05-PRIME/offline_data ~/projects/prime/decider/

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrypoints

### `run.sh`: offline pipeline

Runs the full offline workflow (preprocessing → train → evaluate → online simulation checks). Run from the repo root:

```bash
./run.sh
```

### `prime_core.train`: model training and evaluation

```bash
python -m prime_core.train --cv              # cross-subject k-fold
python -m prime_core.train --train 18 19 …   # pretrain on listed subjects → offline_results/train/
python -m prime_core.train --test 21 22 …    # evaluate (requires pretrained classifier, thus run --train first)
python -m prime_core.train --train-all       # pretrain on all subjects
```

### `prime_core.preprocessing.run_all_subjects`: batch preprocessing

```bash
python -m prime_core.preprocessing.run_all_subjects [--step preprocess|dipole|both] [--subject sub-018] …
```

Reads `offline_data/raw/`, writes processed epochs under `offline_data/processed/`.

### `prime_core.simulate_online`: per-subject online replay

```bash
python -m prime_core.simulate_online <subject_id>
```

Replays one subject trial-by-trial through the online pipeline and checks against `train.py` outputs.
