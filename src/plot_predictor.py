"""Build main Figure 1 as a single native multi-panel figure (not a paste-up).

The sequence-only predictor of the active centromeric HOR array and its functional
validation, panels a-f:
  a-c  active vs inactive windows on the both-class human chromosomes, across the
       two sequence measures (bzip2, Evo 2 surprise) and the accessibility control
       (violin + inner box-and-whisker);
  d    within-chromosome delta (inactive minus active array bzip2), sorted, one dot
       per chromosome, over the corrected 19-chromosome set;
  e    rank of the CENP-A-active array by homogeneity, per chromosome (rank 1 = the
       homogeneity and length rules compared per chromosome;
  f    CENP-A-active vs other arrays, bzip2 (violin + inner box).

Reads only cached tables (active_inactive_windows.csv, array_level.csv), so it is
fully offline and reproducible.
Output: results/figures/fig1_predictor.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")

ACTIVE, INACTIVE = "#0072b2", "#e69f00"   # Okabe-Ito blue / orange (colourblind-safe)
CENPA_RED, OTHER_GREEN = "#d62728", "#2ca02c"


def violin_box(ax, data, labels, colors):
    """Violin (translucent, coloured) with a narrow dark box-and-whisker inside."""
    pos = list(range(1, len(data) + 1))
    vp = ax.violinplot(data, positions=pos, showextrema=False, widths=0.78)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.40); body.set_edgecolor(c)
        body.set_linewidth(1.0)
    ax.boxplot(data, positions=pos, widths=0.12, showfliers=False, showmeans=True,
               patch_artist=True, medianprops=dict(color="white", lw=1.4),
               meanprops=dict(marker="o", mfc="white", mec="#222222", ms=4, mew=0.8),
               boxprops=dict(facecolor="#3a3a3a", edgecolor="#3a3a3a"),
               whiskerprops=dict(color="#3a3a3a", lw=1.1),
               capprops=dict(color="#3a3a3a", lw=1.1))
    ax.set_xticks(pos); ax.set_xticklabels(labels)


def letter(ax, s, x=-0.13):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=14)


def main():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5,
                         "axes.labelsize": 9.5, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9, "legend.fontsize": 8})
    win = pd.read_csv(os.path.join(TAB, "active_inactive_windows.csv"))
    both = [c for c in win.chrom.unique()
            if {"active", "inactive"} <= set(win[win.chrom == c].hor_state)
            and (win[win.chrom == c].hor_state == "inactive").sum() >= 5]
    hs = win[win.chrom.isin(both)]
    al = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv")).dropna(subset=["bz2"])
    arr = al.copy()
    arr["cenpa_active"] = arr.groupby("chrom").cenpa.transform(lambda s: s == s.max())

    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.34,
                          left=0.08, right=0.975, top=0.9, bottom=0.09)

    # ---- a, b, c : window-level active vs inactive across three tracks ----
    specs = [("bz2_bpb", "bzip2 bits/base\n(homogeneity)", "bits/base", None),
             ("evo2_bpb", "Genome foundation model surprise\n(Evo 2)", "bits/base", None),
             ("accessibility", "Fiber-seq accessibility\n(negative control)", "mean accessibility", (0, 6))]
    for k, (m, title, ylab, yl) in enumerate(specs):
        ax = fig.add_subplot(gs[0, k])
        data = [hs[hs.hor_state == s][m].dropna().values for s in ("active", "inactive")]
        violin_box(ax, data, ["active", "inactive"], [ACTIVE, INACTIVE])
        ax.set_title(title); ax.set_ylabel(ylab)
        if yl:
            ax.set_ylim(*yl)
            ax.annotate("axis clipped\n(outliers to ~18)", (0.97, 0.96),
                        xycoords="axes fraction", ha="right", va="top", fontsize=7,
                        color="#777777")
        letter(ax, "abc"[k])

    # ---- d : within-chromosome delta (inactive - active array bzip2), sorted ----
    axd = fig.add_subplot(gs[1, 0])
    d = []
    for ch, g in al.groupby("chrom"):
        a = g[g.censat_active == True].bz2; i = g[g.censat_active == False].bz2
        if len(a) and len(i):
            d.append((ch.replace("chr", ""), i.mean() - a.mean()))
    d.sort(key=lambda r: r[1])
    ypos = np.arange(len(d)); delta = [v for _, v in d]
    cols = [OTHER_GREEN if v > 0 else INACTIVE for v in delta]
    axd.hlines(ypos, 0, delta, color=cols, lw=1.4, alpha=0.7, zorder=1)
    axd.scatter(delta, ypos, color=cols, s=34, zorder=3, edgecolor="white", lw=0.4)
    axd.axvline(0, color="#333333", lw=0.8)
    axd.set_yticks(ypos); axd.set_yticklabels([c for c, _ in d], fontsize=7.5)
    axd.set_ylim(-0.7, len(d) - 0.3)
    axd.set_xlabel("inactive minus active  (bzip2 bits/base)")
    npos = sum(v > 0 for v in delta)
    axd.set_title(f"Within-chromosome bzip2 contrast\n({npos} of {len(d)} positive)")
    axd.text(0.97, 0.04, "positive = active\nmore homogeneous", transform=axd.transAxes,
             ha="right", va="bottom", fontsize=7.5, color="#555555")
    letter(axd, "d", x=-0.18)

    # ---- e : array length against homogeneity, all 81 arrays ----
    # Figure 1 otherwise shows only homogeneity and predictability, so the feature the paper
    # concludes is the operative one had no panel at all. This adds it, and in the same plot
    # shows the coupling that is the central concession to review: length and bzip2 correlate
    # at rho = -0.51, so the two are not independent signals. The separation is visibly
    # cleaner on the length axis than on the homogeneity axis, which is the whole argument.
    axe = fig.add_subplot(gs[1, 1])
    kb = arr.span_bp / 1e3
    axe.scatter(kb[~arr.cenpa_active], arr.bz2[~arr.cenpa_active], s=26, alpha=0.65,
                color=OTHER_GREEN, edgecolor="white", lw=0.4, zorder=2,
                label=f"competing arrays (n={int((~arr.cenpa_active).sum())})")
    axe.scatter(kb[arr.cenpa_active], arr.bz2[arr.cenpa_active], s=52, alpha=0.9,
                color=CENPA_RED, edgecolor="white", lw=0.6, zorder=3,
                label=f"CENP-A-active (n={int(arr.cenpa_active.sum())})")
    # the length threshold that separates the two classes, drawn where it actually falls
    thr = arr.span_bp[arr.cenpa_active].min() / 1e3
    axe.axvline(thr, color="#333333", ls=":", lw=1.0, zorder=1)
    axe.text(thr * 1.15, arr.bz2.max(), f"smallest active\narray, {thr:.0f} kb",
             fontsize=6.2, color="#555555", va="top")
    rho = spearmanr(np.log10(arr.span_bp), arr.bz2)[0]
    axe.set_xscale("log")
    axe.set_xlabel("array span (kb, log scale)")
    axe.set_ylabel("array mean bzip2 bits/base\n(lower = more homogeneous)")
    axe.set_title(f"Array span versus homogeneity\n"
                  f"(Spearman $\\rho$ = {rho:.2f})")
    axe.legend(loc="lower left", fontsize=6.4, framealpha=0.92, handletextpad=0.3)
    letter(axe, "e", x=-0.2)

    # ---- f : functional active vs other arrays, bzip2 (violin + box) ----
    axf = fig.add_subplot(gs[1, 2])
    violin_box(axf, [arr[arr.cenpa_active].bz2.values, arr[~arr.cenpa_active].bz2.values],
               ["CENP-A\nactive", "other"], [CENPA_RED, OTHER_GREEN])
    axf.set_ylabel("array mean bzip2 bits/base")
    try:
        h = pd.read_csv(os.path.join(TAB, "step4_length_vs_homogeneity.csv"))
        h = h[h.panel.str.startswith("(A)")]
        a = h[h.features == "homogeneity only (bz2+H11)"].iloc[0]
        l = h[h.features == "length only"].iloc[0]
        sub = (f"held-out AUROC {a.auroc:.2f} [{a.lo:.2f}, {a.hi:.2f}]\n"
               f"length alone {l.auroc:.2f} [{l.lo:.2f}, {l.hi:.2f}]")
    except Exception:
        sub = ""
    axf.set_title("Array-level bzip2\nby CENP-A status")
    if sub:
        # inside the axes: an annotation below them is clipped by the figure margin
        axf.text(0.5, 0.02, sub, transform=axf.transAxes, ha="center", va="bottom",
                 fontsize=7.5, color="#444444",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    letter(axf, "f")

    fig.suptitle("A sequence-only predictor of the active centromere and its functional validation",
                 fontsize=13, y=0.965)
    fig.savefig(os.path.join(FIG, "fig1_predictor.png"), dpi=150)
    plt.close(fig)
    print(f"Fig 1 OK; both-class {both}; arrays {len(arr)}; delta+ {npos}/{len(d)}; "
          f"rho(len,bz2) {rho:.2f}; active span median {arr.span_bp[arr.cenpa_active].median()/1e3:.0f} kb "
          f"vs {arr.span_bp[~arr.cenpa_active].median()/1e3:.0f} kb")


if __name__ == "__main__":
    main()
