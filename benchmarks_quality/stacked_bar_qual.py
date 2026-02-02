#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


NAME_MAP = {
    "dl": "Damerau-Levenshtein",
    "jw": "Jaro-Winkler",
    "ngram": "NGram",
}


def extract_key_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    for k in NAME_MAP.keys():
        if f"_{k}_" in base:
            return k
    return os.path.splitext(base)[0]


def compute_counts_from_csv(path: str):
    """
    Berechnet TP/FP/TN/FN + ND mit folgender Logik:
    - auto-match: bestmatched vorhanden UND assigned == bestmatched
    - no-decision: bestmatched vorhanden UND assigned != bestmatched
    - auto-non-match: bestmatched fehlt
    """
    df = pd.read_csv(path)

    is_dup = df["is_duplicate"].astype(bool).to_numpy()

    assigned = df["assignedpatient_patientjpaid"].to_numpy()
    best = df["bestmatchedpatient_patientjpaid"].to_numpy()

    best_isna = pd.isna(best)

    auto_match = (~best_isna) & (assigned == best)
    no_decision = (~best_isna) & (assigned != best)
    auto_nonmatch = best_isna

    TP = int((is_dup & auto_match).sum())
    FP = int(((~is_dup) & auto_match).sum())
    FN = int((is_dup & auto_nonmatch).sum())
    TN = int(((~is_dup) & auto_nonmatch).sum())
    ND = int(no_decision.sum())

    return TP, FP, TN, FN, ND


def add_segment_labels(ax, bottoms, values, x_positions, min_height=250):
    for i, (b, v, x) in enumerate(zip(bottoms, values, x_positions)):
        if v >= min_height:
            ax.text(
                x,
                b + v / 2,
                f"{v}",
                ha="center",
                va="center",
                fontsize=9,
            )


def main():
    files = []
    files.extend(glob.glob(os.path.join("eval", "eval_*_10k.csv")))
    files.extend(glob.glob("eval_*_10k.csv"))
    files = sorted(set(files))

    if not files:
        raise SystemExit(
            "Keine CSV-Dateien gefunden. Erwartet z. B. eval/eval_dl_10k.csv oder eval_dl_10k.csv."
        )

    rows = []
    for f in files:
        key = extract_key_from_filename(f)
        name = NAME_MAP.get(key, key)
        TP, FP, TN, FN, ND = compute_counts_from_csv(f)
        rows.append(
            {"Komparator": name, "RP": TP, "FP": FP, "RN": TN, "FN": FN, "PM": ND}
        )

    df = pd.DataFrame(rows).set_index("Komparator")

    df = df.sort_values(["PM", "RP"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    categories = ["RP", "FP", "RN", "FN", "PM"]
    x = range(len(df.index))

    bottom = [0] * len(df.index)

    for cat in categories:
        vals = df[cat].values
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            label=cat,
            width=0.78,
            edgecolor="black",
            linewidth=0.6,
        )
        add_segment_labels(ax, bottom, vals, x_positions=[p.get_x() + p.get_width() / 2 for p in bars])
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_ylim(0, 10000)
    ax.set_ylabel("Anzahl")
    ax.set_xlabel("")
    #ax.set_title("TP/FP/TN/FN und PM pro Komparator (10.000 Datensätze)")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=0)

    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig("stacked_bar_qual.pdf")
    plt.savefig("stacked_bar_qual.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
