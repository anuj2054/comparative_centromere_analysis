"""Test active-versus-competing homogeneity with the chromosome as the unit.

The reported contrast is a Mann-Whitney over 19 CENP-A-active arrays against 62 competing
ones. Those 62 are clustered inside the same 19 chromosomes, so the test treats as
independent observations that are not, and its inferential unit does not match the
paper's own design, which is a within-chromosome ranking problem.

This runs the paired version. For each chromosome, delta is the competing-array bzip2
minus the active-array bzip2, so positive means the active array is the more homogeneous.
Two definitions of the competitor are reported because neither is obviously right: the
median of that chromosome's competing arrays, and its single most homogeneous competitor,
which is the harder test and is what the top-1 ranking rule actually faces.

Sign test and Wilcoxon signed-rank, both two-sided, on 19 paired observations.

Input:  results/tables/array_level_19chrom.csv
Output: results/tables/paired_within_chrom.csv, results/paired_within_chrom_log.txt
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
OUT = []


def log(m=""):
    print(m)
    OUT.append(str(m))


def main():
    a = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
    rows = []
    for ch, g in a.groupby("chrom"):
        act = g.loc[g.cenpa.idxmax()]
        comp = g.drop(act.name)
        if comp.empty:
            continue
        rows.append({"chrom": ch, "n_competing": len(comp),
                     "bz2_active": act.bz2,
                     "bz2_competing_median": comp.bz2.median(),
                     "bz2_competing_best": comp.bz2.min(),
                     "delta_median": comp.bz2.median() - act.bz2,
                     "delta_best": comp.bz2.min() - act.bz2})
    d = pd.DataFrame(rows)

    log(f"Paired within-chromosome homogeneity, n = {len(d)} chromosomes")
    log("delta = competing bzip2 minus active bzip2; positive = active is more homogeneous\n")
    for col, lab in [("delta_median", "median competing array"),
                     ("delta_best", "most homogeneous competitor")]:
        v = d[col].values
        pos = int((v > 0).sum())
        sign = binomtest(pos, len(v), 0.5, alternative="two-sided").pvalue
        w = wilcoxon(v, alternative="two-sided")
        log(f"  vs {lab}")
        log(f"    positive on {pos}/{len(v)} chromosomes, median delta {np.median(v):+.4f} bits/base")
        log(f"    sign test p = {sign:.3g}   Wilcoxon signed-rank W = {w.statistic:.0f}, p = {w.pvalue:.3g}")
    log("\nChromosomes where the active array is NOT the most homogeneous:")
    for _, r in d[d.delta_best < 0].iterrows():
        log(f"  {r.chrom}: active {r.bz2_active:.4f} vs best competitor {r.bz2_competing_best:.4f} "
            f"(delta {r.delta_best:+.4f})")

    d.to_csv(os.path.join(TAB, "paired_within_chrom.csv"), index=False)
    with open(os.path.join(ROOT, "results", "paired_within_chrom_log.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
