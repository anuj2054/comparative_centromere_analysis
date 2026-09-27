"""Does homogeneity carry predictive information beyond array length?

Reviewer 2's first comment asks for length alone, homogeneity alone, and both, in
one cross-validation framework, and says that if homogeneity adds nothing the title
and conclusions must change. This script runs that comparison three ways:

  (A) human arrays, leave-one-chromosome-out, against the functional CENP-A label
  (B) ape arrays, leave-one-chromosome-out, against each species' CenSat active_hor
  (C) a residual test: regress bzip2 on log array length, and ask whether the part
      of bzip2 that is NOT explained by length still predicts activity

(C) is the decisive test. Length and homogeneity are confounded by drive, which
expands and homogenizes the active array together, so their marginal AUROCs cannot
separate them. What can is whether length-independent homogeneity still carries signal.

Output: results/tables/step4_length_vs_homogeneity.csv ; results/length_vs_homogeneity_log.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
RNG = np.random.default_rng(20260818)
NBOOT = 500
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def loco(df, feats, target):
    y, p = [], []
    for ch in df.chrom.unique():
        tr, te = df[df.chrom != ch], df[df.chrom == ch]
        if tr[target].nunique() < 2 or te[target].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[feats].values)
        clf = LogisticRegression(max_iter=3000).fit(
            sc.transform(tr[feats].values), tr[target].astype(int))
        p.extend(clf.predict_proba(sc.transform(te[feats].values))[:, 1])
        y.extend(te[target].astype(int))
    return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan


def loco_preds(df, feats, target):
    """Held-out predictions from ONE pass of leave-one-chromosome-out on the real data."""
    out = []
    for ch in df.chrom.unique():
        tr, te = df[df.chrom != ch], df[df.chrom == ch]
        if tr[target].nunique() < 2 or te[target].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[feats].values)
        m = LogisticRegression(max_iter=5000).fit(
            sc.transform(tr[feats].values), tr[target].astype(int))
        p = m.predict_proba(sc.transform(te[feats].values))[:, 1]
        out += [dict(chrom=ch, y=int(v), p=pi) for v, pi in zip(te[target], p)]
    return pd.DataFrame(out)


def paired_boot(df, sets, target):
    """Cluster bootstrap over chromosomes, all feature sets on the SAME resamples.

    Fit ONCE on the real data, then resample only at the EVALUATION step. The previous
    version resampled chromosomes and then re-ran leave-one-chromosome-out, which puts a
    twice-drawn chromosome in both the training and test folds. That leaks, it narrows
    the interval for the more flexible model, and it was the reason a published paired
    difference appeared to exclude zero when it does not.
    """
    preds = {k: loco_preds(df, f, target) for k, f in sets.items()}
    chroms = preds[list(sets)[0]].chrom.unique()
    keep = {k: [] for k in sets}
    for _ in range(NBOOT):
        pick = RNG.choice(chroms, len(chroms), replace=True)
        vals = {}
        ok = True
        for k, P in preds.items():
            S = pd.concat([P[P.chrom == c] for c in pick])
            if S.y.nunique() < 2:
                ok = False
                break
            vals[k] = roc_auc_score(S.y, S.p)
        if ok:
            for k, v in vals.items():
                keep[k].append(v)
    return {k: np.array(v) for k, v in keep.items()}


def add_residual(df):
    """bzip2 with the length-explained component removed."""
    x = np.log10(df.span_bp.values).reshape(-1, 1)
    df = df.copy()
    df["log_span"] = x.ravel()
    df["bz2_resid"] = df.bz2.values - LinearRegression().fit(x, df.bz2.values).predict(x)
    df["H11_resid"] = df.H11.values - LinearRegression().fit(x, df.H11.values).predict(x)
    return df


SETS = {
    "length only": ["log_span"],
    "homogeneity only (bz2+H11)": ["bz2", "H11"],
    "length + homogeneity": ["log_span", "bz2", "H11"],
    "length-residual homogeneity": ["bz2_resid", "H11_resid"],
    "GC (baseline)": ["gc"],
}
rows = []


def panel(tag, df, target):
    log("\n" + "=" * 78)
    log(f"{tag}   ({len(df)} arrays, {df.chrom.nunique()} chromosomes, target = {target})")
    log("=" * 78)
    r, s = spearmanr(df.log_span, df.bz2)
    log(f"  Spearman(log length, bzip2) = {r:+.3f}  (p = {s:.2g})  "
        f"-- {'longer arrays are MORE homogeneous' if r < 0 else 'longer arrays are LESS homogeneous'}")
    pt = {k: loco(df, f, target) for k, f in SETS.items()}
    bt = paired_boot(df, SETS, target)
    log(f"\n  {'feature set':30s} {'AUROC':>7s}  {'95% CI':>16s}")
    for k in SETS:
        lo, hi = np.percentile(bt[k], [2.5, 97.5]) if len(bt[k]) else (np.nan, np.nan)
        log(f"  {k:30s} {pt[k]:7.3f}  [{lo:6.3f},{hi:6.3f}]")
        rows.append(dict(panel=tag, features=k, auroc=pt[k], lo=lo, hi=hi))
    log("\n  paired differences (same bootstrap resamples):")
    for a, b in [("homogeneity only (bz2+H11)", "length only"),
                 ("length + homogeneity", "length only"),
                 ("length + homogeneity", "homogeneity only (bz2+H11)"),
                 ("length-residual homogeneity", "GC (baseline)")]:
        d = bt[a] - bt[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        log(f"    {a[:26]:26s} - {b[:22]:22s} = {d.mean():+.3f} "
            f"[{lo:+.3f},{hi:+.3f}]  P(>0)={np.mean(d > 0):.2f}")
        rows.append(dict(panel=tag, features=f"DELTA {a} - {b}", auroc=d.mean(), lo=lo, hi=hi))


def main():
    hum = add_residual(pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv")))
    hum["cenpa_active"] = False
    for _, g in hum.groupby("chrom"):
        hum.loc[g.cenpa.idxmax(), "cenpa_active"] = True
    panel("(A) HUMAN, functional CENP-A label", hum, "cenpa_active")

    ape = pd.read_csv(os.path.join(TAB, "ape_expanded_arrays.csv"))
    ape = add_residual(ape[ape.n_win >= 5])
    panel("(B) APES, CenSat active_hor label", ape, "censat_active")

    pd.DataFrame(rows).to_csv(os.path.join(TAB, "step4_length_vs_homogeneity.csv"), index=False)
    with open(os.path.join(ROOT, "results", "length_vs_homogeneity_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
