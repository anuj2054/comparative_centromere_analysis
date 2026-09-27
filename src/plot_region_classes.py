"""Supplementary Fig. S1: complexity metrics by region class, with the H1-vs-R scatter.

This figure previously existed only as a PNG. Its two halves were produced separately by
build_chr21_map.py, as class_boxplots.png and h1_vs_R.png, and combined outside any script,
so it could not be regenerated from the deposit. This rebuilds it in one pass, which
matters because the Data Availability statement promises figure-generation scripts.

  a-d  class violins with inner box-and-whisker for H1, H11, bzip2 and the Entropy-Rank
       Ratio, over coding, unique and satellite 2 kbp windows of chr21
  e    per-base Shannon H1 against R, showing that low-order entropy saturates where the
       shuffle-null measure still separates the classes

Input:  results/tables/class_features.csv
Output: results/figures/figS01_region_class_complexity.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_plotting import violin_box  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
ORDER = ["coding", "unique", "satellite"]
COL = {"coding": "#1f77b4", "unique": "#2ca02c", "satellite": "#d62728"}
PANELS = [("H1", "Shannon H$_1$ (per-base)"),
          ("H11", "Shannon H$_{11}$ (high-order)"),
          ("bz2_bpb", "bzip2 bits/base"),
          ("R_shuffle", "Entropy-Rank Ratio $R$ (shuffle null)")]


def letter(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def main():
    df = pd.read_csv(os.path.join(TAB, "class_features.csv"))
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))

    for ax, (col, title), lab in zip(axes, PANELS, "abcd"):
        violin_box(ax, [df[df.label == o][col].dropna().values for o in ORDER],
                   ORDER, [COL[o] for o in ORDER])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        letter(ax, lab)

    ax = axes[4]
    for lab, g in df.groupby("label"):
        ax.scatter(g.H1, g.R_shuffle, s=16, alpha=0.6, label=lab, color=COL.get(lab))
    ax.set_xlabel("Shannon H$_1$ (per-base entropy), saturates", fontsize=9)
    ax.set_ylabel("Entropy-Rank Ratio $R$, resolves structure", fontsize=9)
    ax.set_title("H$_1$ cannot separate the classes; $R$ does", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    letter(ax, "e")

    fig.suptitle("Complexity metrics by region class (real T2T-CHM13 chr21)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "figS01_region_class_complexity.png")
    fig.savefig(out, dpi=160)
    print(f"Fig S1 OK -> {out}")
    print(f"  {len(df)} fragments: {df.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
