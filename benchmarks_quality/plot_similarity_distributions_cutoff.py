#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

NAME_MAP = {
    "dl": "Damerau-Levenshtein",
    "jw": "Jaro-Winkler",
    "ngram": "NGram",
}

SENTINEL_CUTOFF = -1e307  

SIM_MIN = 0.0
SIM_MAX = 1.0


def extract_key_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    for k in NAME_MAP.keys():
        if f"_{k}_" in base:
            return k
    return os.path.splitext(base)[0]


def load_weights(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    if "bestmatchedweight" not in df.columns:
        raise ValueError(f"Column 'bestmatchedweight' not found in {path}")

    w = pd.to_numeric(df["bestmatchedweight"], errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(w)
        & (w > SENTINEL_CUTOFF)
        & (w >= SIM_MIN)
        & (w <= SIM_MAX)
    )
    return w[mask]


def find_default_files() -> list[str]:
    files: list[str] = []
    files.extend(glob.glob(os.path.join("eval", "eval_*_10k.csv")))
    files.extend(glob.glob("eval_*_10k.csv"))
    return sorted(set(files))


def make_overlay_kde_plot(
    series_dict: dict[str, np.ndarray],
    out_prefix: str = "similarity_kde_overlay",
):

    all_vals = np.concatenate([v for v in series_dict.values() if len(v) > 0])
    if len(all_vals) == 0:
        raise SystemExit("Keine gültigen weights nach Filterung gefunden.")

    xs = np.linspace(SIM_MIN, SIM_MAX, 600)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(SIM_MIN, SIM_MAX)

    for label, vals in series_dict.items():
        if len(vals) < 5 or np.std(vals) == 0:
            continue
        kde = gaussian_kde(vals)
        ys = kde(xs)
        ax.plot(xs, ys, linewidth=2.2, label=label)

    #ax.set_title("Geglättete Verteilungen der Ähnlichkeitswerte (KDE) pro Komparator", pad=10)
    ax.set_xlabel("Ähnlichkeitswert")
    ax.set_ylabel("Wahrscheinlichkeitsdichte")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", f"{out_prefix}.pdf"))
    plt.savefig(os.path.join("plots", f"{out_prefix}.png"), dpi=300)
    plt.close()


def make_individual_histograms(series_dict: dict[str, np.ndarray]):

    for label, vals in series_dict.items():
        if len(vals) == 0:
            continue

        fig, ax = plt.subplots(figsize=(9.2, 4.6))

        vals = vals[(vals >= SIM_MIN) & (vals <= SIM_MAX)]

        ax.hist(
            vals,
            bins=40,
            density=True,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.6,
            label="Histogram (Dichte)",
        )

        if len(vals) >= 5 and np.std(vals) > 0:
            xs = np.linspace(SIM_MIN, SIM_MAX, 400)
            kde = gaussian_kde(vals)
            ax.plot(xs, kde(xs), linewidth=2.0, label="KDE (Glättung)")
            ax.set_xlim(SIM_MIN, SIM_MAX)

        ax.set_title(f"Verteilung der Ähnlichkeitswerte: {label}", pad=10)
        ax.set_xlabel("Ähnlichkeitswert")
        ax.set_ylabel("Wahrscheinlichkeitsdichte")

        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)

        safe_label = label.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join("plots", f"similarity_distribution_{safe_label}_10k_pretty.pdf"))
        plt.savefig(os.path.join("plots", f"similarity_distribution_{safe_label}_10k_pretty.png"), dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Overlay KDE similarity distributions from eval CSVs")
    ap.add_argument("files", nargs="*", help="Eval CSV files (optional). If omitted, uses default globs.")
    args = ap.parse_args()

    files = args.files if args.files else find_default_files()
    if not files:
        raise SystemExit(
            "Keine Eval-CSV-Dateien gefunden. Erwartet z. B. eval/eval_dl_10k.csv oder eval_dl_10k.csv."
        )

    series: dict[str, np.ndarray] = {}

    for f in files:
        key = extract_key_from_filename(f)
        label = NAME_MAP.get(key, key)
        weights = load_weights(f)

        if len(weights) == 0:
            continue

        series[label] = weights

        std = weights.std(ddof=1) if len(weights) > 1 else 0.0
        print(f"{f}")
        print(f"    Valid weights after filtering: n={len(weights)}")
        print(
            f"    min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}, std={std:.6f}"
        )

    if len(series) == 0:
        raise SystemExit("Keine gültige Series gefunden.")

    make_overlay_kde_plot(series)

    if args.also_individual:
        make_individual_histograms(series)


if __name__ == "__main__":
    main()
