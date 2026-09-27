"""Main Figure 2: cross-species contrast and zero-shot transfer, on the CORRECTED set.

The published figure showed 15 ape chromosomes. Those were a hardcoded subset of the
47 that carry both active_hor and dhor on the primary haplotype; build_ape_censat_census.py
is the census that establishes the 47.
This version shows all 47 and, crucially, plots the LENGTH rule beside homogeneity --
across the apes length is the stronger rule (44/47 vs 35/47), and a figure that hid
that would not survive review.

  a-c  array-level bzip2, active vs relict, per species, all eligible chromosomes
  d    zero-shot scatter: most homogeneous active vs most homogeneous relict array,
       one point per chromosome, the 15-chromosome subset ringed for reference
  e    accuracy of each rule, 15-chromosome subset vs all eligible, with bootstrap CIs

Output: results/figures/fig2_crossspecies.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib_plotting import violin_box  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
ACTIVE, INACTIVE = "#0072b2", "#e69f00"
HOM, LEN = "#2ca02c", "#d62728"
SPCOL = {"chimp": "#5b8ff9", "gorilla": "#9270ca", "bonobo": "#f6bd16"}
RNG = np.random.default_rng(20260818)


def letter(ax, s, x=-0.13):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=14)


def wilson(k, n, z=1.96):
    """Wilson score interval. Unlike a bootstrap over observed chromosomes it does not
    collapse when k == n, where resampling can only ever return a perfect score."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ci(v):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return (np.nan, np.nan)
    b = [RNG.choice(v, len(v), replace=True).mean() for _ in range(5000)]
    return np.percentile(b, [2.5, 97.5])


def main():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8})
    # No n_win filter: panel e is scored on the "all arrays" chromosome table, so
    # filtering the array table here would make panels d and e disagree about which
    # chromosomes exist (it silently dropped 20 of 47 in the first draft).
    arr = pd.read_csv(os.path.join(TAB, "ape_expanded_arrays.csv"))
    ch = pd.read_csv(os.path.join(TAB, "ape_expanded_chrom.csv"))
    ch = ch[ch["filter"] == "all arrays"]

    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                          left=0.075, right=0.975, top=0.9, bottom=0.1)

    # ---- a-c: array-level homogeneity contrast per species ----
    for k, sp in enumerate(["chimp", "gorilla", "bonobo"]):
        ax = fig.add_subplot(gs[0, k])
        g = arr[arr.species == sp]
        data = [g[g.censat_active].bz2.values, g[~g.censat_active].bz2.values]
        violin_box(ax, data, ["active_hor", "dhor"], [ACTIVE, INACTIVE])
        ax.set_ylabel("array mean bzip2 bits/base" if k == 0 else "")
        ax.set_title(f"{sp.capitalize()}\n{len(data[0])} active vs {len(data[1])} relict arrays")
        letter(ax, "abc"[k])

    # ---- d: zero-shot scatter, one point per chromosome ----
    axd = fig.add_subplot(gs[1, 0])
    for sp, g in ch.groupby("species"):
        best = []
        for _, r in g.iterrows():
            a = arr[(arr.chrom == r.chrom) & arr.censat_active].bz2
            i = arr[(arr.chrom == r.chrom) & ~arr.censat_active].bz2
            if len(a) and len(i):
                best.append((i.min(), a.min(), bool(r.in_published_set)))
        if not best:
            continue
        x = [b[0] for b in best]; y = [b[1] for b in best]
        pub = [b[2] for b in best]
        axd.scatter(x, y, s=42, color=SPCOL[sp], alpha=0.85, edgecolor="white",
                    lw=0.5, zorder=3, label=f"{sp} (n={len(best)})")
        axd.scatter([x[j] for j in range(len(x)) if pub[j]],
                    [y[j] for j in range(len(y)) if pub[j]],
                    s=130, facecolors="none", edgecolors="#333333", lw=1.0, zorder=4)
    lim = [min(axd.get_xlim()[0], axd.get_ylim()[0]), max(axd.get_xlim()[1], axd.get_ylim()[1])]
    axd.plot(lim, lim, ls="--", color="#333333", lw=0.9, zorder=1)
    axd.fill_between(lim, lim, lim[0], color=HOM, alpha=0.07, zorder=0)
    axd.set_xlim(*lim); axd.set_ylim(*lim)
    axd.set_xlabel("most homogeneous RELICT array (bzip2)")
    axd.set_ylabel("most homogeneous ACTIVE array (bzip2)")
    axd.set_title("Most homogeneous active vs relict array\n(below the diagonal = rule correct)")
    axd.legend(loc="upper left", fontsize=7.5)
    axd.text(0.97, 0.04, "ringed = 15-chromosome subset\n(shown for reference only)",
             transform=axd.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
    letter(axd, "d", x=-0.18)

    # ---- e: rule accuracy by species, as a forest plot ----
    # Wilson score intervals, not bootstrap: resampling observed chromosomes for a perfect
    # score can only return a perfect score, so gorilla's 13/13 would collapse to a point and
    # read as zero uncertainty. No chance line, because the random-pick expectation follows
    # each chromosome's own candidate count and differs by species (0.379, 0.442, 0.378,
    # 0.396); Fig. S12 handles chance properly.
    axe = fig.add_subplot(gs[1, 1])
    NM = {"chimp": "Chimpanzee", "gorilla": "Gorilla", "bonobo": "Bonobo"}
    rows = [(NM[sp], ch[ch.species == sp]) for sp in ["chimp", "gorilla", "bonobo"]]
    rows += [("All eligible", ch)]
    rules = [("longest array", "length_hit", LEN),
             ("CentroSeek", "lr_hit", "#7f7f7f"),
             ("most homogeneous", "homogeneity_hit", HOM)]
    off = [0.24, 0.0, -0.24]
    ypos = np.arange(len(rows))[::-1]
    for (rname, col, colr), dy in zip(rules, off):
        for y, (_, g) in zip(ypos, rows):
            k, n = int(g[col].sum()), len(g)
            lo, hi = wilson(k, n)
            axe.plot([lo, hi], [y + dy, y + dy], color=colr, lw=1.5, alpha=0.55,
                     solid_capstyle="round", zorder=2)
            axe.plot(k / n, y + dy, "o", color=colr, ms=5.5, mec="white", mew=0.7, zorder=3,
                     label=rname if y == ypos[0] else None)
    axe.axhline(ypos[2] - 0.5, color="#cccccc", lw=0.8, zorder=1)
    axe.set_yticks(ypos)
    axe.set_yticklabels([f"{n}\n(n={len(g)})" for n, g in rows], fontsize=7)
    axe.set_xlim(0.22, 1.04)
    axe.set_xticks([0.4, 0.6, 0.8, 1.0])
    axe.set_xlabel("fraction correctly identified")
    axe.set_title("Top-1 identification,\nby species")
    axe.legend(loc="lower left", fontsize=6.2, ncol=1, framealpha=0.9,
               handletextpad=0.3, borderpad=0.35)
    axe.spines[["top", "right"]].set_visible(False)
    letter(axe, "e", x=-0.30)

    # ---- f : active / largest-rival length ratio, all 47 chromosomes ----
    # The context for "44 of 47": most contests are not close. 34 of 47 chromosomes have an
    # active array more than 20x its largest rival, so the length rule is mostly winning
    # lopsided comparisons. The panel shows that distribution and where it inverts.
    axf = fig.add_subplot(gs[1, 2])
    ch = ch.assign(ratio=ch.active_span_kb / ch.relict_span_kb)
    cc = ch.sort_values("ratio").reset_index(drop=True)
    xs = np.arange(len(cc))
    for sp, g in cc.groupby("species"):
        axf.scatter(g.index, g.ratio, s=26, color=SPCOL[sp],
                    alpha=0.85, edgecolor="white", lw=0.4, zorder=3,
                    label=f"{NM[sp]} (n={len(g)})")
    axf.axhline(1, color="#333333", ls="--", lw=1.0, zorder=2)
    axf.axhspan(cc.ratio.min() * 0.6, 1, color=LEN, alpha=0.07, zorder=1)
    axf.set_yscale("log")
    axf.set_ylim(cc.ratio.min() * 0.6, 1600)
    for k in range(len(cc)):
        if cc.ratio[k] <= 1.0:
            nm = cc.chrom[k].split("_")[0]
            axf.annotate(nm, (k, cc.ratio[k]), textcoords="offset points",
                         xytext=(5, -2), fontsize=6.2, color=LEN, ha="left", va="center")
    # CentroSeek correctness is marked on every chromosome, not only on the three rescues:
    # the model recovers the whole sub-parity tail but pays for it on nine lopsided
    # chromosomes (ratios 2.0-150) that the length rule gets right. Marking only the wins
    # would show the trade-off's upside and hide its cost.
    miss = cc[~cc.lr_hit]
    axf.scatter(miss.index, miss.ratio, s=30, marker="x", color="#7f7f7f", lw=1.1,
                zorder=4, label="CentroSeek incorrect")
    axf.text(len(cc) - 1, 1.30, "active array is the longest", ha="right", va="bottom",
             fontsize=6.5, color="#555555")
    axf.text(len(cc) - 1, 0.72, "length rule fails", ha="right", va="top",
             fontsize=6.5, color=LEN)
    axf.text(len(cc) - 1, 0.42, "CentroSeek correct on all three", ha="right", va="top",
             fontsize=6.5, color="#7f7f7f")
    axf.set_xlabel("eligible ape chromosomes, ranked")
    axf.set_ylabel("active array / largest rival (log)")
    axf.set_title("Active array versus\nlargest rival array")
    axf.legend(loc="upper left", fontsize=6.2, framealpha=0.9, handletextpad=0.3)
    axf.spines[["top", "right"]].set_visible(False)
    letter(axf, "f", x=-0.30)

    fig.suptitle("Cross-species array architecture and sequence-only ranking across eligible ape chromosomes",
                 fontsize=13, y=0.965)
    fig.savefig(os.path.join(FIG, "fig2_crossspecies.png"), dpi=150)
    plt.close(fig)
    print(f"Fig 2 OK; arrays {len(arr)}; chromosomes {len(ch)}; "
          f"homogeneity {ch.homogeneity_hit.sum()}/{len(ch)}; length {ch.length_hit.sum()}/{len(ch)}")


if __name__ == "__main__":
    main()
