"""Verify PRIME can read sub-018's data through its own paradigm classes."""
from pathlib import Path
import numpy as np

# Point PRIME at your data BEFORE importing it
import datasets as m
m.DATA_ROOT_PATH = Path("~/prime-data/processed").expanduser()

# Instantiate the TEP dataset + paradigm
dataset  = m.TEPDataset(subject_list=[18])
paradigm = m.TEPParadigm(tmin=-0.5, tmax=-0.020)

X, y, meta = paradigm.get_data(dataset, subjects=[18])

print(f"X shape: {X.shape}")                # expect (~1100, n_ch, 481) — calibration trims ~100
print(f"y shape: {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")
print(f"meta columns: {list(meta.columns)}")
print(f"NaN labels: {np.isnan(y).sum()}")