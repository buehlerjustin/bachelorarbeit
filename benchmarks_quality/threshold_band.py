#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.DataFrame({
        "Komparator": ["Damerau-\nLevenshtein", "Jaro-Winkler", "NGram"],
        "LOWER": [0.6289896764, 0.6770659617, 0.6167748234],
        "UPPER": [0.7386155202, 0.7565518687, 0.7386155202],
        "FN": [16, 12, 20],
    })

    df["Delta"] = df["UPPER"] - df["LOWER"]

    # Sortieren (besserer Komparator oben)
    df = df.sort_values("FN", ascending=True).reset_index(drop=True)

    # Farben passend zum kumulativen Plot (Matplotlib default cycle)
    color_map = {
        "Damerau-\nLevenshtein": "tab:blue",
        "Jaro-Winkler": "tab:orange",
        "NGram": "tab:green",
    }
    bar_colors = [color_map[k] for k in df["Komparator"]]

    # y-Positionen numerisch
    y_pos = list(range(len(df)))

    fig, ax = plt.subplots(figsize=(9, 3.6))

    # Horizontaler Balken von LOWER bis UPPER (Breite = Delta)
    ax.barh(
        y=y_pos,
        width=df["Delta"],
        left=df["LOWER"],
        height=0.28,
        color=bar_colors,        # <- NEU: Balkenfüllung pro Komparator
        alpha=0.85,
        edgecolor="black",       # <- NEU: alle Umrandungen schwarz
        linewidth=1.0,
    )

    # Marker + Text
    for i, r in df.iterrows():
        # Marker farblich passend zum Balken (optional, wirkt stimmiger)
        c = bar_colors[i]
        # ax.plot(r["LOWER"], i, marker="|", markersize=18, mew=2, color=c)
        # ax.plot(r["UPPER"], i, marker="|", markersize=18, mew=2, color=c)

        ax.text(
            r["UPPER"] + 0.006,
            i,
            f"Δ={r['Delta']:.3f}",
            va="center",
            fontsize=11,
        )

    # Achsenbeschriftung
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Komparator"])

    ax.set_xlabel("Ähnlichkeitswert")
    ax.set_ylabel("Komparator")

    # X-Limits automatisch passend
    xmin = float(df["LOWER"].min()) - 0.02
    xmax = float(df["UPPER"].max()) + 0.05
    ax.set_xlim(xmin, xmax)

    # Dezentes Grid & cleaner Look
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("possible_match_zone_bands.pdf")
    plt.savefig("possible_match_zone_bands.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
