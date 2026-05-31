#!/usr/bin/env python3
"""Field-by-field diff between two MNE .fif epoch files.

Usage:
    python compare_fif.py file_a.fif file_b.fif
"""

import sys
import numpy as np
import mne


# Keys in info that are expected to differ between runs even when data is identical
KNOWN_NOISY_INFO_KEYS = {'meas_id', 'file_id', 'meas_date', 'proc_history'}


def _fmt(value, maxlen=120):
    s = repr(value)
    if len(s) > maxlen:
        s = s[:maxlen] + f"... (len={len(s)})"
    return s


def _compare_arrays(name, a, b, diffs):
    if a.shape != b.shape:
        diffs.append(f"{name}: SHAPE differs  a={a.shape}  b={b.shape}")
        return
    if a.dtype != b.dtype:
        diffs.append(f"{name}: DTYPE differs  a={a.dtype}  b={b.dtype}")
    # Treat NaN == NaN for equality purposes
    try:
        if np.array_equal(a, b, equal_nan=True):
            return
    except TypeError:
        if np.array_equal(a, b):
            return
    # Same shape, numeric: report max abs diff and a few example indices
    try:
        absdiff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_d = float(absdiff.max())
        n_diff = int((absdiff > 0).sum())
        diffs.append(
            f"{name}: VALUES differ  shape={a.shape}  "
            f"n_diff_elems={n_diff}/{a.size}  max_abs_diff={max_d:.6g}"
        )
    except Exception:
        diffs.append(f"{name}: VALUES differ (non-numeric)")


def _compare_values(name, a, b, diffs):
    """Generic recursive comparison."""
    if type(a) is not type(b):
        # Allow numeric cross-type compare (e.g. int vs float)
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
                b, (int, float, np.integer, np.floating)):
            if a != b:
                diffs.append(f"{name}: {_fmt(a)}  !=  {_fmt(b)}")
            return
        diffs.append(f"{name}: TYPE differs  a={type(a).__name__}  b={type(b).__name__}  "
                     f"a={_fmt(a)}  b={_fmt(b)}")
        return

    if isinstance(a, np.ndarray):
        _compare_arrays(name, a, b, diffs)
        return

    if isinstance(a, dict):
        keys_a, keys_b = set(a.keys()), set(b.keys())
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        for k in sorted(only_a):
            diffs.append(f"{name}[{k!r}]: only in A  value={_fmt(a[k])}")
        for k in sorted(only_b):
            diffs.append(f"{name}[{k!r}]: only in B  value={_fmt(b[k])}")
        for k in sorted(keys_a & keys_b, key=str):
            _compare_values(f"{name}[{k!r}]", a[k], b[k], diffs)
        return

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{name}: LENGTH differs  a={len(a)}  b={len(b)}")
            return
        for i, (xa, xb) in enumerate(zip(a, b)):
            _compare_values(f"{name}[{i}]", xa, xb, diffs)
        return

    if isinstance(a, float):
        if np.isnan(a) and np.isnan(b):
            return
        if a != b:
            diffs.append(f"{name}: {a!r}  !=  {b!r}")
        return

    if a != b:
        diffs.append(f"{name}: {_fmt(a)}  !=  {_fmt(b)}")


def compare_info(info_a, info_b, diffs):
    keys_a, keys_b = set(info_a.keys()), set(info_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    for k in sorted(only_a):
        diffs.append(f"info[{k!r}]: only in A")
    for k in sorted(only_b):
        diffs.append(f"info[{k!r}]: only in B")
    for k in sorted(keys_a & keys_b):
        marker = " [noisy]" if k in KNOWN_NOISY_INFO_KEYS else ""
        _compare_values(f"info[{k!r}]{marker}", info_a[k], info_b[k], diffs)


def compare_epochs(path_a, path_b):
    print(f"Loading A: {path_a}")
    ep_a = mne.read_epochs(path_a, preload=True, verbose='ERROR')
    print(f"Loading B: {path_b}")
    ep_b = mne.read_epochs(path_b, preload=True, verbose='ERROR')

    diffs = []

    # --- Data ---
    data_a = ep_a.get_data(copy=False)
    data_b = ep_b.get_data(copy=False)
    _compare_arrays("data", data_a, data_b, diffs)

    # --- Top-level scalars / arrays on the Epochs object ---
    for attr in ['tmin', 'tmax', 'baseline', 'event_id', 'selection', 'drop_log',
                 'reject', 'flat', 'reject_tmin', 'reject_tmax',
                 'on_missing', 'metadata']:
        if hasattr(ep_a, attr) and hasattr(ep_b, attr):
            _compare_values(attr, getattr(ep_a, attr), getattr(ep_b, attr), diffs)

    _compare_arrays("events", ep_a.events, ep_b.events, diffs)
    _compare_arrays("times", ep_a.times, ep_b.times, diffs)

    if ep_a.ch_names != ep_b.ch_names:
        diffs.append(f"ch_names differ: A has {len(ep_a.ch_names)}, B has {len(ep_b.ch_names)}")
        only_a = set(ep_a.ch_names) - set(ep_b.ch_names)
        only_b = set(ep_b.ch_names) - set(ep_a.ch_names)
        if only_a:
            diffs.append(f"  channels only in A: {sorted(only_a)}")
        if only_b:
            diffs.append(f"  channels only in B: {sorted(only_b)}")

    # --- Info ---
    compare_info(ep_a.info, ep_b.info, diffs)

    # --- Report ---
    print()
    print("=" * 72)
    if not diffs:
        print("No differences found.")
        return 0

    noisy = [d for d in diffs if '[noisy]' in d]
    real = [d for d in diffs if '[noisy]' not in d]

    print(f"Found {len(diffs)} difference(s):  {len(real)} substantive, "
          f"{len(noisy)} in known-noisy fields (timestamps/IDs/history).")
    print("=" * 72)

    if real:
        print("\n--- Substantive differences ---")
        for d in real:
            print(f"  {d}")

    if noisy:
        print("\n--- Differences in known-noisy fields (expected) ---")
        for d in noisy:
            print(f"  {d}")

    return 1 if real else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(compare_epochs(sys.argv[1], sys.argv[2]))
