"""Supplementary Fig. S12: the two permutation nulls behind the human ranking result.

The manuscript states that the observed AUROC and top-1 accuracy fall outside a null
built by reassigning the active label within each chromosome. That claim is currently a
sentence; this draws the distribution it refers to, which is the more convincing form.

  a  label-permutation null for the LOCO AUROC (10,000 permutations), observed marked
  b  the same for top-1 accuracy
  c  per-chromosome random-pick null for the parameter-free rules, human and ape

Reads results/tables/label_permutation_null.csv (from check_permutation_nulls.py) and
the two master tables.
Output: results/figures/figS12_permutation_null.png
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
NULL = "#9ecae1"; OBS = "#d62728"; RNG = np.random.default_rng(20260909)
OBS_AUC, OBS_TOP1 = 0.9219, 17 / 19


def randpick_null(n_arrays, nperm=200_000):
    p = 1.0 / np.asarray(n_arrays, dtype=float)
    return (RNG.random((nperm, len(p))) < p).sum(axis=1)


def panel(ax, null, obs, xlabel, title, obs_label):
    ax.hist(null, bins=40, color=NULL, edgecolor="white", lw=0.4)
    ax.axvline(obs, color=OBS, lw=2.0, zorder=5)
    ax.annotate(obs_label, xy=(obs, ax.get_ylim()[1] * 0.92),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="top", color=OBS, fontsize=8, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("permutations")
    ax.set_title(title, fontsize=9)


def main():
    d = pd.read_csv(os.path.join(TAB, "label_permutation_null.csv"))
    h = pd.read_csv(os.path.join(TAB, "MASTER_human_19chrom.csv"))
    a = pd.read_csv(os.path.join(TAB, "MASTER_ape_47chrom.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    panel(axes[0], d.perm_auc.values, OBS_AUC, "leave-one-chromosome-out AUROC",
          f"a  label-permutation null, n = {len(d):,}\nnull mean {d.perm_auc.mean():.3f}, "
          f"sd {d.perm_auc.std():.3f}, max {d.perm_auc.max():.3f}",
          f"observed {OBS_AUC:.3f}")
    panel(axes[1], d.perm_top1.values, OBS_TOP1, "top-1 accuracy across 19 chromosomes",
          f"b  same permutations, top-1\nnull mean {d.perm_top1.mean():.3f}",
          f"observed {OBS_TOP1:.3f} (17/19)")

    ax = axes[2]
    for lbl, narr, obs, col in [("human (19 chr)", h.n_arrays.values, int(h.length_hit.sum()), "#4c78a8"),
                                ("ape (47 chr)", a.n_arrays.values, int(a.length_hit.sum()), "#f58518")]:
        null = randpick_null(narr)
        ax.hist(null / len(narr), bins=30, alpha=0.55, color=col, label=f"{lbl} null",
                edgecolor="white", lw=0.3, density=True)
        ax.axvline(obs / len(narr), color=col, lw=2.0)
    ax.set_xlabel("fraction of chromosomes called correctly")
    ax.set_ylabel("density")
    ax.set_title("c  per-chromosome random-pick null\nvertical lines = observed longest-array rule",
                 fontsize=9)
    ax.legend(fontsize=7.5, loc="upper center")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Permutation nulls for the human ranking result and the parameter-free rules",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG, "figS12_permutation_null.png")
    fig.savefig(out, dpi=200)
    print(f"Fig S12 OK -> {out}")
    print(f"  null AUROC mean {d.perm_auc.mean():.3f} sd {d.perm_auc.std():.3f} "
          f"max {d.perm_auc.max():.3f}; observed {OBS_AUC}")


if __name__ == "__main__":
    main()
