"""Supplementary Fig. S2: Evo 2 surprise as an axis separate from compression and R.

Like Fig. S1, this existed only as a PNG. Its halves came from analyze_evo2_chr21.py as
evo2_vs_metrics.png and evo2_class_box.png and were combined outside any script, so the
figure could not be regenerated from the deposit.

  a  Evo 2 next-token surprise against bzip2 compression, per fragment
  b  the same against the Entropy-Rank Ratio
  c  Evo 2 surprise by region class. Evo 2 and compression rank the classes in the same
     order, but separate different pairs: compression nearly ties coding with unique and
     isolates satellite, Evo 2 nearly ties coding with satellite and isolates unique.

Inputs: results/tables/class_features.csv and results/evo2/evo2_class.csv, merged on
(label, frag) exactly as merge_evo2_chr21.merge_class does.
Output: results/figures/figS02_evo2_vs_compression.png
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
EVO = os.path.join(ROOT, "results", "evo2")
FIG = os.path.join(ROOT, "results", "figures")
ORDER = ["coding", "unique", "satellite"]
COL = {"coding": "#1f77b4", "unique": "#2ca02c", "satellite": "#d62728"}


def letter(ax, s):
    ax.text(-0.14, 1.05, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def main():
    feats = pd.read_csv(os.path.join(TAB, "class_features.csv"))
    evo = pd.read_csv(os.path.join(EVO, "evo2_class.csv"))
    df = feats.merge(evo[["label", "frag", "evo2_bpb"]], on=["label", "frag"], how="inner")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (xcol, xlab), lab in zip(
            axes[:2],
            [("bz2_bpb", "bzip2 bits/base (compression complexity)"),
             ("R_shuffle", "Entropy-Rank Ratio $R$ (shuffle null)")], "ab"):
        for l, g in df.groupby("label"):
            ax.scatter(g[xcol], g.evo2_bpb, s=16, alpha=0.6, label=l, color=COL.get(l))
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel("Evo 2 next-token surprise (bits/base)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        letter(ax, lab)
    axes[0].set_title("Evo 2 surprise versus compression", fontsize=10)
    axes[1].set_title("Evo 2 surprise versus $R$", fontsize=10)
    axes[0].legend(fontsize=8)

    ax = axes[2]
    violin_box(ax, [df[df.label == o].evo2_bpb.dropna().values for o in ORDER],
               ORDER, [COL[o] for o in ORDER])
    ax.set_ylabel("Evo 2 next-token surprise (bits/base)", fontsize=9)
    ax.set_title("Evo 2 surprise by class (`evo2_7b`, GPU inference)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    letter(ax, "c")

    fig.suptitle("Evo 2 captures a different axis: coding is compressible but complex, "
                 "yet low surprise", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIG, "figS02_evo2_vs_compression.png")
    fig.savefig(out, dpi=160)
    print(f"Fig S2 OK -> {out}")
    print(f"  merged {len(df)} fragments: {df.label.value_counts().to_dict()}")
    print("  class means (evo2 bits/base): " +
          ", ".join(f"{o} {df[df.label==o].evo2_bpb.mean():.3f}" for o in ORDER))


if __name__ == "__main__":
    main()
