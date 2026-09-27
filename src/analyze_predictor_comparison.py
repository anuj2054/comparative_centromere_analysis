"""Leave-one-chromosome-out engine for the length-vs-homogeneity comparison (human, 19 chromosomes).

New file. Does not modify any existing table or script.

Key differences from src/analyze_length_vs_homogeneity.py:
  1. Length-residualisation (bz2 ~ log10 span) is fit on the TRAINING fold only.
  2. StandardScaler is fit on the TRAINING fold only.
  3. PRIMARY metric is top-1 per-chromosome accuracy (does argmax of predicted
     probability on the held-out chromosome pick the CENP-A-active array?).
  4. The cluster bootstrap resamples chromosomes ONLY at the evaluation step,
     using held-out predictions computed once on the real data.  It does NOT
     resample-then-refit, which would put a duplicated chromosome into both the
     training and the test fold.
Both a LEAKY variant (all preprocessing fit on the full dataset) and a CLEAN
variant are reported so the size of the leakage effect is visible.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
NBOOT = 2000
RNG = np.random.default_rng(20260818)
LOG = []


def log(m=""):
    print(m)
    LOG.append(str(m))


def load():
    d = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
    d["log_span"] = np.log10(d.span_bp.values)
    d["y"] = False
    for _, g in d.groupby("chrom"):
        d.loc[g.cenpa.idxmax(), "y"] = True
    d["y"] = d.y.astype(int)
    return d.reset_index(drop=True)


RESID_SRC = {"bz2_resid": "bz2", "H11_resid": "H11"}


def make_resid(tr, te, cols):
    tr, te = tr.copy(), te.copy()
    for c in cols:
        src = RESID_SRC[c]
        lr = LinearRegression().fit(tr[["log_span"]].values, tr[src].values)
        tr[c] = tr[src].values - lr.predict(tr[["log_span"]].values)
        te[c] = te[src].values - lr.predict(te[["log_span"]].values)
    return tr, te


def add_resid_full(d):
    d = d.copy()
    for c, src in RESID_SRC.items():
        lr = LinearRegression().fit(d[["log_span"]].values, d[src].values)
        d[c] = d[src].values - lr.predict(d[["log_span"]].values)
    return d


def loco_preds(d, feats, leaky):
    rcols = [f for f in feats if f in RESID_SRC]
    if leaky:
        d = add_resid_full(d)
        sc_full = StandardScaler().fit(d[feats].values)
    p = pd.Series(np.nan, index=d.index, dtype=float)
    for ch in d.chrom.unique():
        tr, te = d[d.chrom != ch], d[d.chrom == ch]
        if leaky:
            Xtr, Xte = sc_full.transform(tr[feats].values), sc_full.transform(te[feats].values)
        else:
            if rcols:
                tr, te = make_resid(tr, te, rcols)
            sc = StandardScaler().fit(tr[feats].values)
            Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
        clf = LogisticRegression(max_iter=5000).fit(Xtr, tr.y.values)
        p.loc[te.index] = clf.predict_proba(Xte)[:, 1]
    return p.values


def per_chrom_hits(d, p):
    out = {}
    for ch, g in d.groupby("chrom", sort=False):
        pr = p[g.index.values]
        out[ch] = int(g.y.values[int(np.argmax(pr))] == 1)
    return out


def evaluate(d, p, chroms):
    hits = 0
    ys, ps = [], []
    for ch in chroms:
        g = d[d.chrom == ch]
        pr = p[g.index.values]
        hits += int(g.y.values[int(np.argmax(pr))] == 1)
        ys.append(g.y.values)
        ps.append(pr)
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else np.nan
    return hits / len(chroms), auc


SETS = {
    "A. length only":              ["log_span"],
    "B. homogeneity only":         ["bz2", "H11"],
    "C. length + homogeneity":     ["log_span", "bz2", "H11"],
    "D. current model (bz2,H11)":  ["bz2", "H11"],
    "GC (baseline)":               ["gc"],
    "length-residual homogeneity": ["bz2_resid", "H11_resid"],
}
PAIRS = [("B. homogeneity only", "A. length only"),
         ("C. length + homogeneity", "A. length only"),
         ("C. length + homogeneity", "B. homogeneity only"),
         ("length-residual homogeneity", "GC (baseline)")]


def run(d, leaky):
    chroms = list(d.chrom.unique())
    preds = {k: loco_preds(d, f, leaky) for k, f in SETS.items()}
    point = {k: evaluate(d, preds[k], chroms) for k in SETS}
    hitmap = {k: per_chrom_hits(d, preds[k]) for k in SETS}
    boot = {k: {"top1": [], "auc": []} for k in SETS}
    for _ in range(NBOOT):
        pick = RNG.choice(chroms, len(chroms), replace=True)
        for k in SETS:
            t, a = evaluate(d, preds[k], pick)
            boot[k]["top1"].append(t)
            boot[k]["auc"].append(a)
    boot = {k: {m: np.asarray(v) for m, v in b.items()} for k, b in boot.items()}
    return preds, point, hitmap, boot


def ci(a):
    a = a[~np.isnan(a)]
    return np.percentile(a, [2.5, 97.5]) if len(a) else (np.nan, np.nan)


def main():
    d = load()
    log(f"Human HOR arrays: {len(d)} arrays, {d.chrom.nunique()} chromosomes, "
        f"{int(d.y.sum())} active (1 per chromosome).")
    log(f"Cluster bootstrap: {NBOOT} resamples of the 19 chromosomes, applied to "
        f"held-out predictions computed once (no resample-then-refit).")

    rows, hitrows = [], []
    store = {}
    for leaky in (True, False):
        tag = "leaky" if leaky else "clean"
        preds, point, hitmap, boot = run(d, leaky)
        store[tag] = (point, hitmap, boot)
        log("\n" + "=" * 96)
        log("LEAKY (scaler + residualisation fit on FULL data)" if leaky
            else "CLEAN (scaler + residualisation fit on TRAINING fold only)")
        log("=" * 96)
        log(f"  {'feature set':30s} {'top-1':>7s} {'top-1 95% CI':>18s} {'AUROC':>7s} {'95% CI':>18s}")
        for k in SETS:
            t, a = point[k]
            tl, th = ci(boot[k]["top1"])
            al, ah = ci(boot[k]["auc"])
            log(f"  {k:30s} {int(round(t*19)):3d}/19  [{tl:.3f},{th:.3f}] {a:7.3f}   [{al:6.3f},{ah:6.3f}]")
            rows.append(dict(variant=tag, feature_set=k, features=",".join(SETS[k]),
                             top1_hits=int(round(t * 19)), top1_n=19, top1_acc=round(t, 4),
                             top1_ci_low=round(tl, 4), top1_ci_high=round(th, 4),
                             auroc=round(a, 4), auroc_ci_low=round(al, 4), auroc_ci_high=round(ah, 4)))
        log("\n  paired differences (same bootstrap resamples):")
        for A, B in PAIRS:
            for m, nm in (("auc", "AUROC"), ("top1", "top-1")):
                i = 1 if m == "auc" else 0
                dd = boot[A][m] - boot[B][m]
                dd = dd[~np.isnan(dd)]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                obs = point[A][i] - point[B][i]
                log(f"    {nm:6s} {A[:28]:28s} - {B[:26]:26s} = {obs:+.4f}  "
                    f"boot mean {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  P(<0)={np.mean(dd<0):.3f}")
                rows.append(dict(variant=tag, feature_set=f"DELTA[{m}] {A} - {B}", features="",
                                 top1_hits="", top1_n="",
                                 top1_acc=round(obs, 4) if m == "top1" else "",
                                 top1_ci_low=round(lo, 4) if m == "top1" else "",
                                 top1_ci_high=round(hi, 4) if m == "top1" else "",
                                 auroc=round(obs, 4) if m == "auc" else "",
                                 auroc_ci_low=round(lo, 4) if m == "auc" else "",
                                 auroc_ci_high=round(hi, 4) if m == "auc" else ""))

    log("\n" + "=" * 96)
    log("PER-CHROMOSOME top-1 hit(1)/miss(0), CLEAN variant   [n = arrays on that chromosome]")
    log("=" * 96)
    keys = list(SETS)
    log("  " + f"{'chrom':7s}{'n':>3s}  " + "  ".join(f"{k[:13]:>13s}" for k in keys))
    hm, hml = store["clean"][1], store["leaky"][1]
    for ch in d.chrom.unique():
        n = int((d.chrom == ch).sum())
        log("  " + f"{ch:7s}{n:3d}  " + "  ".join(f"{hm[k][ch]:>13d}" for k in keys))
        hitrows.append(dict(chrom=ch, n_arrays=n,
                            **{f"clean_{k}": hm[k][ch] for k in keys},
                            **{f"leaky_{k}": hml[k][ch] for k in keys}))
    log("  " + f"{'TOTAL':7s}{len(d):3d}  " + "  ".join(f"{sum(hm[k].values()):>13d}" for k in keys))

    pd.DataFrame(rows).to_csv(os.path.join(TAB, "PREDICTOR_comparison.csv"), index=False)
    pd.DataFrame(hitrows).to_csv(os.path.join(TAB, "PREDICTOR_per_chrom_hits.csv"), index=False)
    with open(os.path.join(ROOT, "results", "PREDICTOR_audit_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log("\nwrote results/tables/PREDICTOR_comparison.csv")


if __name__ == "__main__":
    main()
