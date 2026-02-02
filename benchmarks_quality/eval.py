#!/usr/bin/env python3
"""
- Ground truth column: `is_duplicate`
- Prediction columns:
  - `bestmatchedpatient_patientjpaid` (NaN if no candidate / no decision)
  - `assignedpatient_patientjpaid` (final assigned patient id)

Berechnet TP/FP/TN/FN + ND mit folgender Logik::
- auto-match: bestmatchedpatient_patientjpaid is NOT NaN AND assigned == bestmatched
- no-decision (possible match): bestmatchedpatient_patientjpaid is NOT NaN AND assigned != bestmatched
- auto-non-match: bestmatchedpatient_patientjpaid IS NaN
"""

import glob
import os
from typing import Tuple

import pandas as pd


DEFAULT_GLOBS = [
    os.path.join("eval", "eval_*_10k.csv"),
    "eval_*_10k.csv",
]


def metrics_from_eval_csv(path: str) -> Tuple[int, int, int, int, int, float, float, float, int, int]:
    df = pd.read_csv(path)

    # Ground truth
    is_dup = df["is_duplicate"].astype(bool).to_numpy()

    assigned = df["assignedpatient_patientjpaid"].to_numpy()
    best = df["bestmatchedpatient_patientjpaid"].to_numpy()

    best_isna = pd.isna(best)

    auto_match = (~best_isna) & (assigned == best)
    no_decision = (~best_isna) & (assigned != best)
    auto_nonmatch = best_isna

    assert int(auto_match.sum() + no_decision.sum() + auto_nonmatch.sum()) == len(df)

    decided = ~no_decision

    TP = int((is_dup & auto_match).sum())
    FP = int(((~is_dup) & auto_match).sum())
    FN = int((is_dup & auto_nonmatch).sum())
    TN = int(((~is_dup) & auto_nonmatch).sum())

    N_total = len(df)
    N_no_decision = int(no_decision.sum())
    N_decided = int(decided.sum())

    assert TP + FP + FN + TN == N_decided, "TP/FP/FN/TN must sum to decided cases"

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return TP, FP, TN, FN, N_no_decision, precision, recall, f1, N_total, N_decided


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate RL CSVs (TP/FP/TN/FN + no-decision)")
    ap.add_argument("files", nargs="*", help="CSV files to evaluate (optional; otherwise uses default globs)")
    args = ap.parse_args()

    if args.files:
        files = args.files
    else:
        files = []
        for g in DEFAULT_GLOBS:
            files.extend(glob.glob(g))
        files = sorted(set(files))

    if not files:
        raise SystemExit(
            "No files found. Pass CSVs as arguments or place them in ../eval/ as eval_*_10k.csv."
        )

    for f in files:
        TP, FP, TN, FN, ND, p, r, f1, N_total, N_decided = metrics_from_eval_csv(f)
        print(f"{f}")
        print(f"    Total N={N_total}")
        print(f"    No-decision (possible) ND={ND}")
        print(f"    Decided N={N_decided}")
        print(f"    TP={TP} FP={FP} TN={TN} FN={FN} (sum decided={TP+FP+TN+FN}={N_decided})")
        print(f"    Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
