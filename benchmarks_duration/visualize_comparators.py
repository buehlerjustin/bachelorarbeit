#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Style / Helpers ---------------------------

def set_paper_style():
    # Einheitlicher Stil
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
    })


def clean_axes(ax):
    """Entfernt unnötige Rahmenlinien und setzt Grid unter die Daten."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def savefig(basepath: str, formats=("png", "pdf")) -> None:
    """Speichert als PNG + PDF (PDF ist ideal für LaTeX)."""
    plt.tight_layout()
    for ext in formats:
        plt.savefig(f"{basepath}.{ext}")
    plt.close()


@dataclass
class SeriesData:
    name: str
    df_all: pd.DataFrame
    df_ok: pd.DataFrame
    duration_col: str
    status_col: Optional[str]
    ts_col: Optional[str]


def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    key = key.lower()
    for c in df.columns:
        if key in c.lower():
            return c
    return None


def load_csv(path: str, name: str, only_2xx: bool) -> SeriesData:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    duration_col = _find_col(df, "duration")
    status_col = _find_col(df, "status")
    ts_col = _find_col(df, "timestamp")

    if duration_col is None:
        raise ValueError(f"[{name}] Keine 'duration'-Spalte gefunden. Spalten: {list(df.columns)}")

    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df_all = df[df[duration_col].notna()].copy()

    if only_2xx and status_col is not None:
        df_ok = df_all[df_all[status_col].between(200, 299, inclusive="both")].copy()
    else:
        df_ok = df_all.copy()

    if ts_col is not None:
        df_all[ts_col] = pd.to_numeric(df_all[ts_col], errors="coerce")
        df_ok[ts_col] = pd.to_numeric(df_ok[ts_col], errors="coerce")

    return SeriesData(name, df_all, df_ok, duration_col, status_col, ts_col)


def durations(series: SeriesData) -> np.ndarray:
    return series.df_ok[series.duration_col].to_numpy()


# --------------------------- Plot 1: kumultativ  ---------------------------

def plot_cumulative(series_list, outdir, prefix):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    end_points = [] 

    for s in series_list:
        d = durations(s)
        y_s = np.cumsum(d) / 1000.0  # ms -> s
        x = np.arange(1, len(y_s) + 1)

        ax.plot(x, y_s, linewidth=2.0, label=s.name)

        if len(x) > 0:
            end_points.append((float(y_s[-1]), float(x[-1]), f"{float(y_s[-1]):.0f} s"))

    end_points.sort(key=lambda t: t[0])

    min_gap = 40.0
    y_last = -1e18

    for y_end, x_end, txt in end_points:
        y_label = y_end
        if y_label - y_last < min_gap:
            y_label = y_last + min_gap
        y_last = y_label

        ax.text(
            x_end * 1.005,
            y_label,
            txt,
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Anzahl der Einträge (kumulativ)")
    ax.set_ylabel("Kumulierte Dauer (s)")
    #ax.set_title("Kumulierte Laufzeit über die Anzahl der Einträge")

    ax.margins(x=0.10)

    ax.legend(loc="upper left", frameon=True)
    clean_axes(ax)

    savefig(os.path.join(outdir, f"{prefix}01_kumulativ"))


# --------------------------- ECDF helpers ---------------------------

def ecdf_xy(d: np.ndarray):
    d = np.sort(d)
    y = np.arange(1, len(d) + 1) / len(d)
    return d, y


def compute_tail_metrics(series_list: List[SeriesData], p_zoom: float) -> pd.DataFrame:
    rows = []
    for s in series_list:
        total = len(s.df_all)
        ok = len(s.df_ok)
        err_rate = (1 - ok / total) if total else np.nan

        d = durations(s)
        if len(d) == 0:
            rows.append({
                "Comparator": s.name,
                "n_total": total,
                "n_used": ok,
                "err_rate": err_rate,
                "p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "p999": np.nan,
                f"p{p_zoom:g}": np.nan,
                "max": np.nan,
                "count_gt_p99": np.nan,
            })
            continue

        p50, p90, p95, p99, p999 = np.percentile(d, [50, 90, 95, 99, 99.9])
        pz = np.percentile(d, p_zoom)
        mx = float(np.max(d))
        count_gt_p99 = int(np.sum(d > p99))

        rows.append({
            "Comparator": s.name,
            "n_total": total,
            "n_used": ok,
            "err_rate": err_rate,
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "p999": float(p999),
            f"p{p_zoom:g}": float(pz),
            "max": mx,
            "count_gt_p99": count_gt_p99,
        })

    return pd.DataFrame(rows)


def plot_ecdf_zoom(series_list: List[SeriesData], outdir: str, prefix: str, p_zoom: float) -> float:
    pvals = []
    for s in series_list:
        d = durations(s)
        if len(d) > 0:
            pvals.append(np.percentile(d, p_zoom))
    xmax = float(max(pvals)) if pvals else 1.0

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for s in series_list:
        d = durations(s)
        if len(d) == 0:
            continue
        x, y = ecdf_xy(d)
        ax.step(x, y, where="post", linewidth=2.0, label=s.name)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("Dauer (ms)")
    ax.set_ylabel("Anteil ≤ x")
    #ax.set_title(f"ECDF der Dauer (Zoom bis P{p_zoom:g})")

    ax.legend(loc="lower right", frameon=True)
    clean_axes(ax)

    savefig(os.path.join(outdir, f"{prefix}02_ecdf_zoom"))
    return xmax


def plot_tail_ccdf(series_list, outdir, prefix, x_from_ms=None, use_pivot_percentile=99.0):

    if x_from_ms is None:
        pvals = []
        for s in series_list:
            d = durations(s)
            if len(d) > 0:
                pvals.append(np.percentile(d, use_pivot_percentile))
        x_from_ms = float(max(pvals)) if pvals else 0.0

    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    global_min_y = 1.0
    global_xmax = 0.0

    for s in series_list:
        d = durations(s)
        if len(d) == 0:
            continue

        x = np.sort(d)
        y = np.arange(1, len(x) + 1) / len(x)
        ccdf = 1.0 - y

        mask = x >= x_from_ms
        if np.any(mask):
            ax.step(x[mask], ccdf[mask], where="post", linewidth=2.0, label=s.name)

            global_min_y = min(global_min_y, 1.0 / len(x))

            thr = 10.0 / len(x)
            valid = ccdf[mask] >= thr
            if np.any(valid):
                global_xmax = max(global_xmax, float(x[mask][valid][-1]))
            else:
                global_xmax = max(global_xmax, float(x[mask][-1]))

    ax.set_xlabel("Dauer (ms)")
    ax.set_ylabel("P(Dauer > x)")
    ax.set_yscale("log")

    ax.set_ylim(max(global_min_y * 0.8, 1e-6), 1)

    if global_xmax > 0:
        ax.set_xlim(x_from_ms, global_xmax * 1.1)

    ax.axvline(x_from_ms, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(x_from_ms, 0.9, f"Start (P{use_pivot_percentile:g})",
            rotation=90, va="top", ha="right", fontsize=9)

    ax.legend(loc="upper right", frameon=True)
    clean_axes(ax)
    savefig(os.path.join(outdir, f"{prefix}05_tail_ccdf"))

import numpy as np
import pandas as pd

def tail_table(durations_by_name: dict[str, np.ndarray], p=99.0, global_threshold=True):
    # durations_by_name: {"backendA": array([...]), "backendB": array([...])}

    if global_threshold:
        all_vals = np.concatenate([v for v in durations_by_name.values() if len(v) > 0])
        u = np.percentile(all_vals, p) if len(all_vals) else np.nan
        thresholds = {k: u for k in durations_by_name.keys()}
    else:
        thresholds = {k: (np.percentile(v, p) if len(v) else np.nan) for k, v in durations_by_name.items()}

    rows = []
    for name, x in durations_by_name.items():
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        n = len(x)
        u = thresholds[name]
        tail = x[x > u] if n and np.isfinite(u) else np.array([])
        n_tail = len(tail)

        row = {
            "name": name,
            "n_total": n,
            "p99_u_ms": float(u) if np.isfinite(u) else np.nan,
            "n_tail": n_tail,
            "tail_share_%": (100.0 * n_tail / n) if n else 0.0,
            "ES99_mean_tail_ms": float(np.mean(tail)) if n_tail else np.nan,
            "median_tail_ms": float(np.median(tail)) if n_tail else np.nan,
            "p95_tail_ms": float(np.percentile(tail, 95)) if n_tail else np.nan,
            "max_ms": float(np.max(x)) if n else np.nan,
            "mean_excess_ms": float(np.mean(tail - u)) if n_tail else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ES99_mean_tail_ms", ascending=False)
    return df



# --------------------------- Plot 3: percentiles ---------------------------

def plot_percentiles(series_list: List[SeriesData], outdir: str, prefix: str,
                     percentiles=(50, 90, 95, 99)) -> None:
    rows = []
    for s in series_list:
        d = durations(s)
        vals = np.percentile(d, percentiles) if len(d) else [np.nan] * len(percentiles)
        rows.append({"Komparator": s.name, **{f"P{p}": float(v) for p, v in zip(percentiles, vals)}})

    dfp = pd.DataFrame(rows)
    
    print("\n=== Duration-Perzentile (ms) ===")
    for _, row in dfp.iterrows():
        print(
            f"{row['Komparator']}: "
            f"P50={row['P50']:.3f} ms, "
            f"P90={row['P90']:.3f} ms, "
            f"P95={row['P95']:.3f} ms, "
            f"P99={row['P99']:.3f} ms"
        )

    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    x = np.arange(len(dfp))
    width = 0.18
    offsets = np.linspace(-width*1.5, width*1.5, len(percentiles))

    for off, p in zip(offsets, percentiles):
        ax.bar(x + off, dfp[f"P{p}"], width=width, edgecolor="black", linewidth=0.5, label=f"P{p}")

    ax.set_xticks(x)
    ax.set_xticklabels(dfp["Komparator"], rotation=0)
    ax.set_xlabel("Komparator")
    ax.set_ylabel("Dauer (ms)")
    #ax.set_title("Perzentile der Dauer (P50/P90/P95/P99)")
    ax.legend(loc="upper left", frameon=True, ncol=2)

    clean_axes(ax)
    savefig(os.path.join(outdir, f"{prefix}03_perzentile"))

    dfp.to_csv(os.path.join(outdir, f"{prefix}perzentile.csv"), index=False)


# --------------------------- Plot 4: time series (binned) ---------------------------

def plot_duration_trend_binned(series_list: List[SeriesData], outdir: str, prefix: str,
                               bin_size: int = 500,
                               y_limit_mode: str = "p99",
                               y_margin: float = 0.10):

    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(9.2, 3.3*n), sharex=True)
    if n == 1:
        axes = [axes]

    global_p99_max = None
    if y_limit_mode == "p99":
        p99_max_list = []
        for s in series_list:
            df = s.df_ok.copy()
            if len(df) == 0:
                continue
            if "#" in df.columns:
                df = df.sort_values("#")

            d = df[s.duration_col].to_numpy()
            bin_id = (np.arange(len(d)) // bin_size)
            g = pd.DataFrame({"d": d, "bin": bin_id}).groupby("bin")["d"]
            p99 = g.quantile(0.99).to_numpy()
            if len(p99):
                p99_max_list.append(np.nanmax(p99))

        if p99_max_list:
            global_p99_max = float(np.nanmax(p99_max_list)) * (1.0 + y_margin)

    for ax, s in zip(axes, series_list):
        df = s.df_ok.copy()
        if len(df) == 0:
            clean_axes(ax)
            continue

        if "#" in df.columns:
            df = df.sort_values("#")

        d = df[s.duration_col].to_numpy()
        x = np.arange(1, len(d) + 1)

        bin_id = (np.arange(len(d)) // bin_size)
        tmp = pd.DataFrame({"x": x, "d": d, "bin": bin_id})
        g = tmp.groupby("bin")["d"]

        med = g.median().to_numpy()
        p95 = g.quantile(0.95).to_numpy()
        p99 = g.quantile(0.99).to_numpy()

        bin_index = g.count().index.to_numpy()
        x_bin = (bin_index * bin_size) + (bin_size / 2)

        ax.fill_between(x_bin, med, p95, alpha=0.18, label="Band: P50..P95")
        ax.plot(x_bin, med, linewidth=2.0, label="P50 (Median)")
        ax.plot(x_bin, p95, linewidth=2.0, label="P95")
        ax.plot(x_bin, p99, linewidth=2.0, label="P99")

        ax.set_ylabel("Dauer (ms)")
        clean_axes(ax)

        ax.legend(loc="upper left", frameon=True)

        if y_limit_mode == "p99" and global_p99_max is not None:
            ax.set_ylim(0, global_p99_max)

    axes[-1].set_xlabel("Request #")

    plt.tight_layout()
    savefig(os.path.join(outdir, f"{prefix}04_trend_binned_p99"))


# --------------------------- Main ---------------------------

def main():
    set_paper_style()

    ap = argparse.ArgumentParser(
        description="Visualisierungen (Kumulativ, ECDF-Zoom, Optional Tail-CCDF, Perzentile, Zeitverlauf)"
    )
    ap.add_argument("--damerau", default="./results/DamerauLevenshteinComparator_result.csv")
    ap.add_argument("--jaro", default="./results/JaroWinklerComparator_result.csv")
    ap.add_argument("--qgram", default="./results/q-gram_result.csv")

    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--prefix", default="")

    ap.add_argument("--include-non2xx", action="store_true",
                    help="Wenn gesetzt: Non-2xx werden NICHT herausgefiltert (falls Status-Spalte existiert).")

    ap.add_argument("--ecdf-zoom-p", type=float, default=99.0,
                    help="ECDF Zoom-Grenze als Perzentil (z.B. 99, 99.5, 99.9).")

    ap.add_argument("--tail-ccdf", action="store_true",
                    help="Erzeugt zusätzlich einen Tail-Plot (CCDF) ab der ECDF-Zoom-Grenze.")

    ap.add_argument("--bin-size", type=int, default=500,
                    help="Bin-Groesse für Trend-Plot (Requests pro Bin).")

    args = ap.parse_args()
    only_2xx = not args.include_non2xx

    ensure_outdir(args.outdir)

    series_list = [
        load_csv(args.damerau, "Damerau-Levenshtein", only_2xx=only_2xx),
        load_csv(args.jaro, "Jaro-Winkler", only_2xx=only_2xx),
        load_csv(args.qgram, "NGram", only_2xx=only_2xx),
    ]

    # Plots
    plot_cumulative(series_list, args.outdir, args.prefix)
    xmax_zoom = plot_ecdf_zoom(series_list, args.outdir, args.prefix, p_zoom=args.ecdf_zoom_p)
    plot_percentiles(series_list, args.outdir, args.prefix)
    plot_duration_trend_binned(series_list, args.outdir, args.prefix, bin_size=args.bin_size)
    plot_tail_ccdf(
    series_list,
    args.outdir,
    args.prefix,
    x_from_ms=None,
    use_pivot_percentile=99.0
)
    tail_df = compute_tail_metrics(series_list, p_zoom=args.ecdf_zoom_p)
    tail_csv = os.path.join(args.outdir, f"{args.prefix}tail_metrics.csv")
    tail_df.to_csv(tail_csv, index=False)

    if args.tail_ccdf:
        plot_tail_ccdf(series_list, args.outdir, args.prefix, x_from=xmax_zoom)

    print(f"Fertig. Output in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
