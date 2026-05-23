# analyze_preft.py
# Usage: python3 analyze_preft.py path/to/results_summary.csv

import csv
import statistics
import sys

if len(sys.argv) != 2:
    print("Usage: python3 analyze_preft.py path/to/results_summary.csv")
    sys.exit(1)

rows = []
with open(sys.argv[1], 'r') as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"Number of (subject, fold) rows: {len(rows)}")
print()

cols = {
    'pre_calib_zero_shot_roc_auc_all':     'PRE-ZS  ROC AUC (all)    ',
    'pre_calib_zero_shot_roc_auc_extreme': 'PRE-ZS  ROC AUC (extreme)',
    'post_calib_zero_shot_roc_auc_all':    'PRE-CAL ROC AUC (all)    ',
    'post_calib_zero_shot_roc_auc_extreme':'PRE-CAL ROC AUC (extreme)',
    'finetuned_roc_auc_all':               'PRE-FT  ROC AUC (all)    ',
    'finetuned_roc_auc_extreme':           'PRE-FT  ROC AUC (extreme)',
}

print(f"{'Condition':<28} {'median':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
print("-" * 72)
for col, label in cols.items():
    vals = [float(r[col]) for r in rows if r.get(col)]
    if not vals:
        continue
    median = statistics.median(vals)
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{label:<28} {median:>8.4f} {mean:>8.4f} {stdev:>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

print()
print("Paper targets: PRE-FT median = 0.68 (all), 0.77 (extreme)")
print()
print("Sorted PRE-FT ROC AUC (all):")
vals = sorted(float(r['finetuned_roc_auc_all']) for r in rows)
for v in vals:
    bar = "█" * max(0, int((v - 0.5) * 100))
    print(f"  {v:.3f} {bar}")
