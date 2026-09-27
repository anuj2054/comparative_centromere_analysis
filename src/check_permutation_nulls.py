"""The two significance tests behind the human ranking claims.

Neither appeared in any draft before the revision.

TEST A. Per-chromosome hit rates against the chance baseline. The text states a
"chance baseline near 40 per cent" and reports 19/19, 17/19 (human) and 44/47,
38/47, 35/47 (ape) against it, but no test is performed. Chance is not a single
number: a chromosome with two candidate arrays gives 1/2 and one with eleven
gives 1/11, so the null is a per-chromosome random pick weighted by that
chromosome's own array count. Monte Carlo over that null gives both the correct
expected accuracy and an exact-style p-value.

TEST B. Label-permutation null for the predictor AUROC. Permuting labels freely
would destroy the design, because exactly one array per chromosome is active.
The permutation therefore reassigns the active label WITHIN each chromosome,
preserving one active per chromosome and the array-count structure, then reruns
the full leave-one-chromosome-out fit. This asks the right question: given that
some array on each chromosome is active, does sequence identify WHICH one?

The machinery is imported from analyze_predictor_comparison so the null and the reported
estimate come from identical code rather than a reimplementation.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_predictor_comparison import load, loco_preds, evaluate  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
NPERM = 10_000
RNG = np.random.default_rng(20260907)
OUT = []


def log(m=""):
    print(m)
    OUT.append(str(m))


# ------------------------------------------------------------------ test A --
def hit_rate_test(n_arrays, observed, label, nperm=200_000):
    """P(>= observed hits) when each chromosome's call is a uniform random array."""
    n_arrays = np.asarray(n_arrays, dtype=float)
    p_each = 1.0 / n_arrays
    exp = p_each.sum()
    draws = RNG.random((nperm, len(n_arrays))) < p_each
    null = draws.sum(axis=1)
    p = (np.sum(null >= observed) + 1) / (nperm + 1)
    log(f"  {label:28s} observed {observed:3d}/{len(n_arrays):<3d}"
        f"   chance {exp/len(n_arrays):6.1%} (expected {exp:5.2f} hits)"
        f"   null mean {null.mean():5.2f}   p = {p:.3g}"
        + ("  (< 1/nperm)" if p <= 1.0 / (nperm + 1) else ""))
    return p


# ------------------------------------------------------------------ test B --
def permute_within_chrom(d):
    """Move the active label to a uniformly random array on the same chromosome."""
    y = np.zeros(len(d), dtype=int)
    for _, g in d.groupby("chrom", sort=False):
        y[RNG.choice(g.index.values)] = 1
    return y


def label_permutation_null(d, feats, obs_auc, obs_top1, nperm=NPERM):
    chroms = list(d.chrom.unique())
    aucs, tops = [], []
    for i in range(nperm):
        dd = d.copy()
        dd["y"] = permute_within_chrom(dd)
        p = loco_preds(dd, feats, leaky=False)
        t, a = evaluate(dd, p, chroms)
        aucs.append(a)
        tops.append(t)
        if (i + 1) % 250 == 0:
            log(f"    ... {i+1}/{nperm} permutations")
    aucs, tops = np.array(aucs), np.array(tops)
    p_auc = (np.sum(aucs >= obs_auc) + 1) / (nperm + 1)
    p_top = (np.sum(tops >= obs_top1) + 1) / (nperm + 1)
    log(f"  null AUROC  mean {aucs.mean():.3f}  sd {aucs.std():.3f}  "
        f"95th pct {np.percentile(aucs,95):.3f}  max {aucs.max():.3f}")
    log(f"  observed AUROC {obs_auc:.3f}  ->  empirical p = {p_auc:.3g}")
    log(f"  null top-1  mean {tops.mean():.3f}  95th pct {np.percentile(tops,95):.3f}")
    log(f"  observed top-1 {obs_top1:.3f}  ->  empirical p = {p_top:.3g}")
    return aucs, tops, p_auc, p_top


def main():
    log("TEST A. per-chromosome hit rates vs a per-chromosome random-pick null")
    log()
    h = pd.read_csv(os.path.join(TAB, "MASTER_human_19chrom.csv"))
    log(f"human, {len(h)} chromosomes, arrays per chromosome "
        f"{h.n_arrays.min()}-{h.n_arrays.max()} (median {int(h.n_arrays.median())})")
    hit_rate_test(h.n_arrays, int(h.length_hit.sum()), "human longest-array")
    hit_rate_test(h.n_arrays, int(h.homogeneity_hit.sum()), "human most-homogeneous")
    hit_rate_test(h.n_arrays, int(h.model_hit.sum()), "human CentroSeek")
    log()
    a = pd.read_csv(os.path.join(TAB, "MASTER_ape_47chrom.csv"))
    log(f"ape, {len(a)} chromosomes, arrays per chromosome "
        f"{a.n_arrays.min()}-{a.n_arrays.max()} (median {int(a.n_arrays.median())})")
    hit_rate_test(a.n_arrays, int(a.length_hit.sum()), "ape longest-array")
    hit_rate_test(a.n_arrays, int(a.model_hit.sum()), "ape CentroSeek")
    hit_rate_test(a.n_arrays, int(a.homogeneity_hit.sum()), "ape most-homogeneous")

    log()
    log("TEST B. label-permutation null for the human predictor (bz2 + H11), LOCO")
    log()
    d = load()
    chroms = list(d.chrom.unique())
    obs_top1, obs_auc = evaluate(d, loco_preds(d, ["bz2", "H11"], leaky=False), chroms)
    log(f"  observed: AUROC {obs_auc:.4f}, top-1 {obs_top1:.4f} "
        f"({int(round(obs_top1*len(chroms)))}/{len(chroms)})")
    aucs, tops, p_auc, p_top = label_permutation_null(
        d, ["bz2", "H11"], obs_auc, obs_top1)
    pd.DataFrame({"perm_auc": aucs, "perm_top1": tops}).to_csv(
        os.path.join(TAB, "label_permutation_null.csv"), index=False)
    log()
    log(f"  wrote {os.path.join(TAB, 'label_permutation_null.csv')}")

    with open(os.path.join(ROOT, "results", "permutation_log.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
