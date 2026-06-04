# Online PRIME

## Setting up online pipeline

In NeuroSimo, create a project named `prime` (this creates the directory `~/projects/prime`). After that, run:

```bash
cd ~/projects/prime
rm -rf decider
git clone https://github.com/okahilak/prime.git decider

# Assuming Tübingen's samba share is mounted at ~/samba
rsync -ah --info=progress2 ~/samba/Projects/2026-05-PRIME/offline_data ~/projects/prime/decider/

cd ~/projects/prime/decider

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m prime_core.preprocessing.build_fsaverage
```

## Entrypoints

### `run.sh`: offline pipeline

Runs the full offline workflow (preprocessing -> train -> test -> trial-by-trial testing).

Run in the `~/projects/prime/decider` directory:

```bash
./run.sh
```

### `run_all_subjects.py`: batch preprocessing

```bash
python -m prime_core.preprocessing.run_all_subjects
```

Reads `offline_data/raw/`, writes processed epochs under `offline_data/processed/`.

After finished, check that the cross-validation results are as expected, by running:

```
python3 -m prime_core.analyze_results offline_results/cv/results_summary.csv
```

### `train.py`: model training and evaluation

```bash
# cross-subject k-fold
python -m prime_core.train --cv

# pretrain on listed subjects, saved to offline_results/train/
python -m prime_core.train --train 18 19

# test on listed subjects (requires pretrained classifier - run first with --train argument)
python -m prime_core.train --test 21 22

# pretrain on all subjects
python -m prime_core.train --train-all
```

### `test_by_trial.py`: testing trial-by-trial

```bash
python -m prime_core.test_by_trial <subject_id>
```

Replays one subject trial-by-trial and checks against `train.py --test` batch testing outputs.

This script is one step closer to the online pipeline than `train.py --test`. Its predictions and
labels should therefore match the corresponding batch testing outputs.

## Testing NeuroSimo integration with offline data

Create csv-formatted simulator dataset for subject 21 (only for the first 200 trials to make it faster):

```bash
python -m prime_core.tools.create_simulator_data 21 --short
```

Make it available to NeuroSimo:

```bash
cp -r ~/projects/prime/decider/offline_data/simulator/sub-021/ ~/projects/prime/eeg_simulator/
```

Open NeuroSimo, select the project `prime`, and switch the dataset to `sub-021-short.json`.

### Test online simulation using events

Switch the decider to `simulate_by_events.py`. To run faster, set "Playback speed" to a higher value
(e.g. 4.0).

Start the session and wait for the calibration trials to finish.

After the calibration trials, predictions and labels start printing to the UI.
Check that they match the outputs of `python -m prime_core.test_by_trial 21`.

### Test online simulation using periodic processing

Switch the decider to `simulate_by_periodic.py`. Note that with periodic processing, high
playback speeds may make the system to fall behind. On the BNPLAB NeuroSimo computer, a playback
speed of 0.5 is recommended.

Start the session and wait for the calibration trials to finish.

After the calibration trials, predictions and labels start printing to the UI.
Check that they match the outputs of `python -m prime_core.test_by_trial 21`.

## Running online pipeline in NeuroSimo

Train a classifier using all subjects:

```bash
python3 -m prime_core.train --train-all
```

After that, copy the protocol file to the NeuroSimo protocols directory:

```bash
cd ~/projects/prime/decider
cp protocol.yaml ../protocols
```

Select the project `prime`. Switch the decider to `prime.py`, and protocol to `protocol.yaml`.
Start EEG streaming from the device, and start the session.
