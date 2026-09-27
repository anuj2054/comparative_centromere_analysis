"""Supplementary Fig. S13: the resolution floor of array-mean bzip2, and chromosome 3.

The manuscript declines to call chromosome 3 either way, on the grounds that the two
contenders are separated by less than the variation an array's own mean shows when the
window grid is shifted. That is currently asserted; this shows it.

  a  distribution of phase spread across the 81 filtered arrays, with the chr3 gap marked
  b  the chr3 contest across the eight grid offsets, where the ranking reverses

Reads results/tables/window_phase_sensitivity.csv (from check_window_phase.py).
Output: results/figures/figS13_window_phase.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
A_COL, B_COL, GAP = "#4c78a8", "#f58518", "#d62728"
A, B = "hor_3_2(S01/1C3H1L)", "hor_3_3(S01/1C3H1L)"


def main():
    df = pd.read_csv(os.path.join(TAB, "window_phase_sensitivity.csv"))
    offs = sorted(int(c.replace("bz2_off", "")) for c in df.columns if c.startswith("bz2_off"))
    c3 = df[df.chrom == "chr3"].set_index("array_name")
    va = np.array([c3.loc[A, f"bz2_off{o}"] for o in offs])
    vb = np.array([c3.loc[B, f"bz2_off{o}"] for o in offs])
    gap = abs(va[0] - vb[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

    ax = axes[0]
    ax.hist(df.phase_spread, bins=30, color=A_COL, edgecolor="white", lw=0.4, alpha=0.85)
    med = df.phase_spread.median()
    ax.axvline(med, color="#333333", ls="--", lw=1.2)
    ax.axvline(gap, color=GAP, lw=2.0)
    ax.annotate(f"median spread\n{med:.4f}", xy=(med, ax.get_ylim()[1] * 0.78),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color="#333333")
    ax.annotate(f"chr3 gap\n{gap:.4f}", xy=(gap, ax.get_ylim()[1] * 0.45),
                xytext=(8, 0), textcoords="offset points", fontsize=8,
                color=GAP, fontweight="bold")
    ax.set_xlabel("spread in array-mean bzip2 across 8 window-grid offsets (bits/base)")
    ax.set_ylabel("arrays")
    ax.set_title(f"a  resolution floor across the {len(df)} filtered arrays\n"
                 f"max {df.phase_spread.max():.3f} bits/base", fontsize=9)

    ax = axes[1]
    ax.plot(offs, va, "-o", color=A_COL, lw=1.8, ms=6, label="hor_3_2  (CENP-A-richest, 858 kbp)")
    ax.plot(offs, vb, "-o", color=B_COL, lw=1.8, ms=6, label="hor_3_3  (relict, 34 kbp)")
    for o, x, y in zip(offs, va, vb):
        lo, hi = (min(x, y), max(x, y))
        ax.fill_between([o - 90, o + 90], lo, hi,
                        color=(A_COL if x < y else B_COL), alpha=0.18, lw=0)
    wins = int((va < vb).sum())
    ax.set_xticks(offs)
    ax.set_xlabel("window-grid offset (bp)")
    ax.set_ylabel("array mean bzip2 (bits/base)")
    ax.set_title(f"b  chromosome 3, ranking reverses with grid phase\n"
                 f"lower = more homogeneous; hor_3_2 leads at {wins} of {len(offs)} offsets",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Array-mean homogeneity has a finite resolution set by window placement",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "figS13_window_phase.png")
    fig.savefig(out, dpi=200)
    print(f"Fig S13 OK -> {out}")
    print(f"  median spread {med:.4f}, max {df.phase_spread.max():.4f}, "
          f"chr3 gap {gap:.4f}, hor_3_2 wins {wins}/{len(offs)}")


if __name__ == "__main__":
    main()
