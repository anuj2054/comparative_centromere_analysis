"""Is the HOR repeat PERIOD a length-independent signal?

The scale sweep threw up an unplanned observation: the median dominant
self-similarity period is ~680 bp in active arrays and ~172 bp in inactive relicts,
and 171 bp is the alpha-satellite monomer. That suggests relicts have decayed toward
monomeric periodicity while live arrays retain a long higher-order unit.

If period predicts activity AFTER length is removed, it is the length-independent
signal that bzip2 failed to provide in the length-vs-homogeneity comparison -- which is what Reviewer 2's first
comment demands.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
sw = pd.read_csv(os.path.join(TAB, "step6_scale_sweep.csv"))
per = sw.groupby(["chrom", "name"], as_index=False).period.first()
a = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv")).merge(
    per, on=["chrom", "name"], how="inner").dropna(subset=["period"])
a["cenpa_active"] = False
for _, g in a.groupby("chrom"):
    a.loc[g.cenpa.idxmax(), "cenpa_active"] = True
a["log_span"] = np.log10(a.span_bp)
a["log_period"] = np.log10(a.period)
x = a.log_span.values.reshape(-1, 1)
a["period_resid"] = a.log_period - LinearRegression().fit(x, a.log_period).predict(x)

print(f"arrays with a period estimate: {len(a)} "
      f"({a.censat_active.sum()} CenSat-active)\n")
for lab, col in [("CenSat", "censat_active"), ("CENP-A", "cenpa_active")]:
    A, I = a.period[a[col]], a.period[~a[col]]
    print(f"{lab:7s} label: median period active {A.median():6.0f} bp  "
          f"inactive {I.median():6.0f} bp   MWU p={mannwhitneyu(A,I).pvalue:.2g}")
r, p = spearmanr(a.log_span, a.log_period)
print(f"\nSpearman(log length, log period) = {r:+.3f} (p={p:.2g})   "
      f"-- compare bzip2's {-0.513:+.3f}")


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


SETS = {"length only": ["log_span"],
        "period only": ["log_period"],
        "length-residual period": ["period_resid"],
        "bzip2+H11 (homogeneity)": ["bz2", "H11"],
        "length + period": ["log_span", "log_period"],
        "length + period + bz2 + H11": ["log_span", "log_period", "bz2", "H11"]}
RNG = np.random.default_rng(20260818)
chroms = a.chrom.unique()
boot = {k: [] for k in SETS}
for _ in range(500):
    pick = RNG.choice(chroms, len(chroms), replace=True)
    d = pd.concat([a[a.chrom == c].assign(chrom=f"{c}__{i}")
                   for i, c in enumerate(pick)], ignore_index=True)
    v = {k: loco(d, f, "cenpa_active") for k, f in SETS.items()}
    if not any(np.isnan(x) for x in v.values()):
        for k in SETS:
            boot[k].append(v[k])

print(f"\n{'feature set':30s} {'AUROC':>7s} {'95% CI':>18s}   (target = CENP-A label)")
pt = {}
for k, f in SETS.items():
    pt[k] = loco(a, f, "cenpa_active")
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    print(f"{k:30s} {pt[k]:7.3f}  [{lo:6.3f},{hi:6.3f}]")
print("\npaired differences:")
for x_, y_ in [("period only", "length only"),
               ("length + period", "length only"),
               ("length-residual period", "length only"),
               ("length + period + bz2 + H11", "length + period")]:
    d = np.array(boot[x_]) - np.array(boot[y_])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {x_[:27]:27s} - {y_[:17]:17s} = {d.mean():+.3f} "
          f"[{lo:+.3f},{hi:+.3f}]  P(>0)={np.mean(d>0):.2f}")
