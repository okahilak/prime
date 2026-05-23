#!/usr/bin/env python3
"""
Compare two .npz files field by field on their common top-level keys.

Designed for verifying that refactored dipole fitting matches the original
(e.g. response_extraction_info.npz vs fitted_dipoles.npz, which only overlap
on trial_dipoles_fixed_ori / trial_dipoles_free_ori).

Usage:
    python compare_npz.py REF.npz NEW.npz [--rtol 1e-6] [--atol 1e-10]
    python compare_npz.py REF.npz NEW.npz --keys trial_dipoles_fixed_ori trial_dipoles_free_ori
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# Defaults: refactor changes floating-point order (batched SVD, different
# accumulation), so bit-identity is not expected. ~1e-9 relative is typical.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-10


class Report:
    def __init__(self):
        self.ok = []
        self.warn = []
        self.fail = []

    def pass_(self, path, detail=""):
        self.ok.append((path, detail))

    def warn_(self, path, detail):
        self.warn.append((path, detail))

    def fail_(self, path, detail):
        self.fail.append((path, detail))

    def summary(self):
        print()
        print("=" * 70)
        print(f"PASS: {len(self.ok)}   WARN: {len(self.warn)}   FAIL: {len(self.fail)}")
        print("=" * 70)
        if self.warn:
            print("\nWARNINGS:")
            for p, d in self.warn:
                print(f"  ! {p}: {d}")
        if self.fail:
            print("\nFAILURES:")
            for p, d in self.fail:
                print(f"  X {p}: {d}")
        else:
            print("\nNo failures. Refactor looks equivalent within tolerance.")


def load(path):
    """Load an npz saved with **dict; values come back as 0-d object arrays."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]
            for k in z.files}


def fmt_diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    # avoid div by zero
    rel = np.where(scale > 0, diff / np.maximum(scale, 1e-300), 0.0)
    return (f"max_abs={np.max(diff):.3e}  max_rel={np.max(rel):.3e}  "
            f"shape={a.shape}")


def compare_array(path, a, b, rtol, atol, report):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        report.fail_(path, f"shape mismatch: {a.shape} vs {b.shape}")
        return
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            report.pass_(path, fmt_diff(a, b) if a.size else "empty")
        else:
            report.fail_(path, fmt_diff(a, b))
    elif a.dtype == object or b.dtype == object:
        if a.size == 0:
            report.pass_(path, "empty object array")
            return
        flat_a, flat_b = a.ravel(), b.ravel()
        if all(isinstance(x, dict) for x in flat_a) and all(
            isinstance(x, dict) for x in flat_b
        ):
            compare_list_of_dicts(
                path, flat_a.tolist(), flat_b.tolist(), rtol, atol, report
            )
            return
        try:
            arr_a = np.asarray(flat_a).reshape(a.shape)
            arr_b = np.asarray(flat_b).reshape(b.shape)
            if arr_a.dtype != object and arr_b.dtype != object:
                compare_array(path, arr_a, arr_b, rtol, atol, report)
                return
        except (ValueError, TypeError):
            pass
        for i, (xa, xb) in enumerate(zip(flat_a, flat_b)):
            compare(f"{path}[{i}]", xa, xb, rtol, atol, report)
    else:
        try:
            equal = np.array_equal(a, b)
        except (ValueError, TypeError):
            for i, (xa, xb) in enumerate(zip(a.ravel(), b.ravel())):
                compare(f"{path}[{i}]", xa, xb, rtol, atol, report)
            return
        if equal:
            report.pass_(path, f"exact match ({a.shape}, {a.dtype})")
        else:
            report.fail_(path, f"non-numeric arrays differ ({a.shape}, {a.dtype})")


def compare_scalar(path, a, b, rtol, atol, report):
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        if a == b:
            report.pass_(path, f"int {a}")
        else:
            report.fail_(path, f"int differs: {a} vs {b}")
        return
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            report.pass_(path, f"float diff={abs(a-b):.3e}")
        else:
            report.fail_(path, f"float differs: {a} vs {b}  diff={abs(a-b):.3e}")
        return
    if a == b:
        report.pass_(path, f"{type(a).__name__} match")
    else:
        report.fail_(path, f"{type(a).__name__} differs: {a!r} vs {b!r}")


def compare_dict(path, a, b, rtol, atol, report, strict_keys=False):
    ka, kb = set(a.keys()), set(b.keys())
    if strict_keys and ka != kb:
        report.fail_(path, f"key sets differ: only-ref={kb-ka} only-new={ka-kb}")
        return
    for k in sorted(ka & kb):
        compare(f"{path}.{k}", a[k], b[k], rtol, atol, report, strict_keys=strict_keys)


def compare_list_of_dicts(path, a, b, rtol, atol, report):
    if len(a) != len(b):
        report.fail_(path, f"list length differs: {len(a)} vs {len(b)}")
        return
    if len(a) == 0:
        report.pass_(path, "empty list")
        return
    # Aggregate per-key across the list — much more readable than per-index spam.
    keys = set(a[0].keys()) & set(b[0].keys())
    for k in sorted(keys):
        sub_a = [item[k] for item in a]
        sub_b = [item[k] for item in b]
        # Try to compare as a stacked array when possible
        try:
            arr_a = np.array(sub_a)
            arr_b = np.array(sub_b)
            if arr_a.dtype != object and arr_b.dtype != object:
                compare_array(f"{path}[*].{k}", arr_a, arr_b, rtol, atol, report)
                continue
        except (ValueError, TypeError):
            pass
        # Fall back to element-wise; only report first failure per key to limit noise.
        first_fail = None
        for i, (xa, xb) in enumerate(zip(sub_a, sub_b)):
            sub_report = Report()
            compare(f"{path}[{i}].{k}", xa, xb, rtol, atol, sub_report)
            if sub_report.fail:
                first_fail = sub_report.fail[0]
                break
        if first_fail is None:
            report.pass_(f"{path}[*].{k}", f"all {len(a)} elements match")
        else:
            report.fail_(f"{path}[*].{k}", f"first diff at {first_fail[0]}: {first_fail[1]}")


def compare_dataframe(path, a, b, rtol, atol, report):
    if a.shape != b.shape:
        report.fail_(path, f"DataFrame shape differs: {a.shape} vs {b.shape}")
        return
    if list(a.columns) != list(b.columns):
        report.warn_(path, f"columns differ: {list(a.columns)} vs {list(b.columns)}")
    for col in [c for c in a.columns if c in b.columns]:
        sa, sb = a[col].to_numpy(), b[col].to_numpy()
        compare_array(f"{path}[{col!r}]", sa, sb, rtol, atol, report)


def compare(path, a, b, rtol, atol, report, strict_keys=False):
    # Unwrap 0-d object arrays
    if isinstance(a, np.ndarray) and a.dtype == object and a.shape == ():
        a = a.item()
    if isinstance(b, np.ndarray) and b.dtype == object and b.shape == ():
        b = b.item()

    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        compare_dataframe(path, a, b, rtol, atol, report)
    elif isinstance(a, dict) and isinstance(b, dict):
        compare_dict(path, a, b, rtol, atol, report, strict_keys=strict_keys)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) > 0 and isinstance(a[0], dict):
            compare_list_of_dicts(path, a, b, rtol, atol, report)
        else:
            # Treat as array if possible
            try:
                arr_a, arr_b = np.array(a), np.array(b)
                if arr_a.dtype != object:
                    compare_array(path, arr_a, arr_b, rtol, atol, report)
                    return
            except (ValueError, TypeError):
                pass
            if len(a) != len(b):
                report.fail_(path, f"length differs: {len(a)} vs {len(b)}")
                return
            for i, (xa, xb) in enumerate(zip(a, b)):
                compare(f"{path}[{i}]", xa, xb, rtol, atol, report)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        compare_array(path, a, b, rtol, atol, report)
    elif isinstance(a, (int, float, np.integer, np.floating, str, bool, np.bool_)):
        compare_scalar(path, a, b, rtol, atol, report)
    elif a is None and b is None:
        report.pass_(path, "both None")
    else:
        # Unknown type — try equality, fall back to repr.
        try:
            if a == b:
                report.pass_(path, f"{type(a).__name__} match (eq)")
            else:
                report.fail_(path, f"{type(a).__name__} differ: {a!r} vs {b!r}")
        except Exception as e:
            report.warn_(path, f"could not compare {type(a).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--show-all", action="store_true",
                   help="Print every comparison (default: only warnings + failures)")
    p.add_argument("--keys", nargs="+", metavar="KEY",
                   help="Top-level keys to compare (default: intersection of both files)")
    p.add_argument("--strict-keys", action="store_true",
                   help="Fail if top-level or nested dict key sets differ")
    args = p.parse_args()

    print(f"Loading reference: {args.reference}")
    ref = load(args.reference)
    print(f"Loading new:       {args.new}")
    new = load(args.new)
    print(f"Tolerances: rtol={args.rtol}  atol={args.atol}")

    ka, kb = set(ref.keys()), set(new.keys())
    if args.keys:
        keys = sorted(set(args.keys))
        missing_ref = set(keys) - ka
        missing_new = set(keys) - kb
        if missing_ref or missing_new:
            raise SystemExit(
                f"Requested keys not found: "
                f"only in ref missing={sorted(missing_ref)} "
                f"only in new missing={sorted(missing_new)}"
            )
    else:
        keys = sorted(ka & kb)

    only_ref = ka - set(keys)
    only_new = kb - set(keys)
    if only_ref or only_new:
        print(f"Skipping keys only in reference ({len(only_ref)}): {sorted(only_ref)}")
        print(f"Skipping keys only in new ({len(only_new)}): {sorted(only_new)}")
    if not keys:
        raise SystemExit("No keys to compare (no overlap and --keys not given).")
    print(f"Comparing {len(keys)} key(s): {keys}")
    print()

    report = Report()
    if args.strict_keys and (ka != kb or set(keys) != ka or set(keys) != kb):
        report.fail_("<top>", f"top-level keys differ: only-ref={ka-kb} only-new={kb-ka}")
    for k in keys:
        compare(k, ref[k], new[k], args.rtol, args.atol, report, strict_keys=args.strict_keys)

    if args.show_all:
        print("ALL CHECKS:")
        for path, detail in report.ok:
            print(f"  ✓ {path}: {detail}")

    report.summary()
    raise SystemExit(0 if not report.fail else 1)


if __name__ == "__main__":
    main()
