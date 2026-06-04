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

python -m prime_core.preprocessing.build_fsaverage
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

### `prime_core.test_by_trial`: per-subject online replay

```bash
python -m prime_core.test_by_trial <subject_id>
```

Replays one subject trial-by-trial and checks against `train.py --test` batch testing outputs.

## Testing NeuroSimo integration

Create simulator data for subject 21:

```bash
python -m prime_core.tools.create_simulator_data 21 --short
```

Make it available to NeuroSimo:

```bash
cp -r ~/projects/prime/decider/offline_data/simulator/sub-021/ ~/projects/prime/eeg_simulator/
```

Open NeuroSimo, select the project `prime`, and switch the dataset to `sub-021-short.json`. Switch
the decider to `simulate_by_events.py`. Start the session.
