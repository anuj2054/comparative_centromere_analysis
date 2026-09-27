"""Biology-first analysis: do ACTIVE centromeric alpha-satellite HOR arrays
(where the kinetochore assembles) differ in sequence-information and chromatin
state from INACTIVE relict HOR arrays on the same chromosomes?

Active vs inactive is taken from CenSat v2.1 HOR names: the live/active array
carries an 'L' designation (e.g. hor_21_3(S2C13/21H1L)); inactive arrays do not
(e.g. hor_21_1(S4/6C13/14/21H1)). This is independent of our metrics.

Windows: chr21 map (with real Evo 2) + 5 centromere regions (chr13/14/15/22/X).
We label each window active/inactive by midpoint overlap with HOR intervals, then
compare accessibility and complexity, per chromosome and pooled.

Outputs: results/tables/active_vs_inactive_hor.csv
         results/figures/active_vs_inactive_hor.png
         results/active_centromere_log.txt
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
CEN = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/censat_v2.1.bb"
UCSC = {"chr13": "chr13", "chr14": "chr14", "chr15": "chr15", "chr21": "chr21",
        "chr22": "chr22", "chrX": "chrX"}
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def hor_intervals(bb, ch):
    """Return (active_spans, inactive_spans) for chromosome ch."""
    ents = bb.entries(ch, 0, bb.chroms()[ch]) or []
    active, inactive = [], []
    for s, e, r in ents:
        name = r.split("\t")[0]
        if name.lower().startswith("hor"):
            (active if re.search(r"H\d+L", name) else inactive).append((s, e))
    return active, inactive


def label_active(mid, active, inactive):
    for s, e in active:
        if s <= mid < e:
            return "active"
    for s, e in inactive:
        if s <= mid < e:
            return "inactive"
    return "other"


def load_pooled():
    multi = pd.read_csv(os.path.join(TAB, "multichrom_windows.csv"))
    # merge in real Evo 2 surprise for the 5 new chromosomes (by chrom+start)
    ev = pd.read_csv(os.path.join(ROOT, "results", "evo2", "evo2_multichrom.csv"))
    multi = multi.merge(ev[["chrom", "start", "evo2_bpb"]], on=["chrom", "start"],
                        how="left")
    chr21 = pd.read_csv(os.path.join(TAB, "map_windows_functional.csv")).copy()
    chr21["chrom"] = "chr21"
    cols = ["chrom", "start", "end", "gc", "H11", "bz2_bpb", "R_shuffle",
            "accessibility", "evo2_bpb"]
    return pd.concat([multi[cols], chr21[cols]], ignore_index=True)


def main():
    df = load_pooled()
    bb = pyBigWig.open(CEN)
    labels = []
    for ch in df.chrom.unique():
        active, inactive = hor_intervals(bb, UCSC[ch])
        sub = df[df.chrom == ch]
        for _, r in sub.iterrows():
            mid = (int(r.start) + int(r.end)) // 2 - 1
            labels.append(label_active(mid, active, inactive))
    bb.close()
    df["hor_state"] = labels
    hor = df[df.hor_state.isin(["active", "inactive"])].copy()

    log("== Active vs inactive centromeric HOR: window counts ==")
    log(pd.crosstab(hor.chrom, hor.hor_state).to_string())

    metrics = ["accessibility", "bz2_bpb", "R_shuffle", "H11", "evo2_bpb", "gc"]
    log("\n== Pooled means (active vs inactive HOR) ==")
    a = hor[hor.hor_state == "active"]
    i = hor[hor.hor_state == "inactive"]
    rows = []
    for m in metrics:
        av, iv = a[m].dropna(), i[m].dropna()
        try:
            U, p = mannwhitneyu(av, iv, alternative="two-sided")
        except ValueError:
            p = float("nan")
        y = np.r_[np.ones(len(av)), np.zeros(len(iv))]
        x = np.r_[av.values, iv.values]
        auc = max(roc_auc_score(y, x), 1 - roc_auc_score(y, x)) if len(iv) > 2 else float("nan")
        rows.append({"metric": m, "active_mean": av.mean(), "inactive_mean": iv.mean(),
                     "mannwhitney_p": p, "auroc": auc})
        log(f"  {m:14s} active {av.mean():7.3f}  inactive {iv.mean():7.3f}  "
            f"p={p:.1e}  AUROC={auc:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "active_vs_inactive_hor.csv"), index=False)

    # Evo 2 surprise active vs inactive, per chromosome (now on all 6 chroms)
    log("\n== Per-chromosome Evo 2 surprise: active vs inactive ==")
    for ch, g in hor.groupby("chrom"):
        ga = g[g.hor_state == "active"].evo2_bpb.dropna()
        gi = g[g.hor_state == "inactive"].evo2_bpb.dropna()
        if len(gi) >= 5:
            log(f"  {ch}: active {ga.mean():.3f} (n={len(ga)})  "
                f"inactive {gi.mean():.3f} (n={len(gi)})  "
                f"delta={ga.mean()-gi.mean():+.3f}")

    # per-chromosome compression consistency (the homogeneity signal)
    log("\n== Per-chromosome bz2 (homogeneity): active vs inactive ==")
    for ch, g in hor.groupby("chrom"):
        ga = g[g.hor_state == "active"].bz2_bpb
        gi = g[g.hor_state == "inactive"].bz2_bpb
        if len(gi) >= 5:
            log(f"  {ch}: active {ga.mean():.3f} (n={len(ga)})  "
                f"inactive {gi.mean():.3f} (n={len(gi)})  "
                f"delta={ga.mean()-gi.mean():+.3f}")

    # Stratified test on chromosomes that contain BOTH classes (chr14/15/22),
    # avoiding the Simpson's-paradox that pooling all 6 induces for Evo 2
    # (chrX/chr13 are active-only at a higher Evo 2 baseline).
    both = [ch for ch in hor.chrom.unique()
            if {"active", "inactive"} <= set(hor[hor.chrom == ch].hor_state)
            and (hor[hor.chrom == ch].hor_state == "inactive").sum() >= 5]
    hs = hor[hor.chrom.isin(both)]
    log(f"\n== Stratified test on both-class chromosomes {both} ==")
    for m in ["bz2_bpb", "evo2_bpb", "H11"]:
        va = hs[hs.hor_state == "active"][m].dropna()
        vi = hs[hs.hor_state == "inactive"][m].dropna()
        U, p = mannwhitneyu(va, vi, alternative="two-sided")
        y = np.r_[np.ones(len(va)), np.zeros(len(vi))]
        x = np.r_[va.values, vi.values]
        auc = max(roc_auc_score(y, x), 1 - roc_auc_score(y, x))
        log(f"  {m:10s} active {va.mean():.3f}  inactive {vi.mean():.3f}  "
            f"p={p:.1e}  AUROC={auc:.3f}")

    # GC-confound control: does compression separate active/inactive beyond GC?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    log("\n== GC-confound control (5-fold CV AUROC) ==")
    y = (hor.hor_state == "active").astype(int).values
    for feats in [["gc"], ["bz2_bpb"], ["gc", "bz2_bpb"], ["gc", "H11"]]:
        X = StandardScaler().fit_transform(hor[feats].values)
        auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5,
                              scoring="roc_auc").mean()
        log(f"  {'+'.join(feats):16s} AUROC = {max(auc, 1-auc):.3f}")

    # GC-matched test (restrict to the overlapping GC band)
    band = hor[(hor.gc >= 0.34) & (hor.gc <= 0.40)]
    ba = band[band.hor_state == "active"].bz2_bpb
    bi = band[band.hor_state == "inactive"].bz2_bpb
    if len(bi) > 5:
        U, p = mannwhitneyu(ba, bi, alternative="two-sided")
        log(f"\n  GC-matched (0.34-0.40) bz2: active {ba.mean():.3f} (n={len(ba)})  "
            f"inactive {bi.mean():.3f} (n={len(bi)})  p={p:.1e}")

    # cache the labeled per-window table so combined figures can be rebuilt offline
    # H11 and the window coordinates are deposited too. Table 1 reports an H11 row for
    # the chr14/15/22 panel, and without these columns that row cannot be reproduced from
    # the deposit, nor can any window be traced back to its position.
    hor[["chrom", "start", "end", "hor_state", "bz2_bpb", "H11", "evo2_bpb",
         "accessibility", "gc"]].to_csv(
        os.path.join(TAB, "active_inactive_windows.csv"), index=False)

    figures(hor)
    with open(os.path.join(ROOT, "results", "active_centromere_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("\nDone.")


def figures(hor):
    # restrict to chromosomes carrying BOTH array types (chr14/15/22) so the
    # plotted contrast matches the within-chromosome statistics (pooling all 6
    # induces Simpson's paradox for Evo 2 via chromosome-specific baselines).
    both = [ch for ch in hor.chrom.unique()
            if {"active", "inactive"} <= set(hor[hor.chrom == ch].hor_state)
            and (hor[hor.chrom == ch].hor_state == "inactive").sum() >= 5]
    hs = hor[hor.chrom.isin(both)]
    metrics = ["bz2_bpb", "evo2_bpb", "accessibility"]
    titles = ["bzip2 bits/base (homogeneity)", "Evo 2 surprise (predictability)",
              "Fiber-seq accessibility"]
    ylims = [None, None, (0, 6)]   # clip accessibility outliers (~18) so boxes are visible
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for k, (ax, m, t, yl) in enumerate(zip(axes, metrics, titles, ylims)):
        data = [hs[hs.hor_state == s][m].dropna().values for s in ["active", "inactive"]]
        ax.boxplot(data, tick_labels=["active", "inactive"], showmeans=True)
        ax.set_title(t, fontsize=10)
        if yl:
            ax.set_ylim(*yl)
            ax.set_xlabel("(axis clipped, outliers to ~18)", fontsize=7)
        ax.text(-0.08, 1.02, "abc"[k], transform=ax.transAxes, fontweight="bold", fontsize=12)
    fig.suptitle(f"Active vs inactive centromeric HOR arrays ({'/'.join(both)})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "active_vs_inactive_hor.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
