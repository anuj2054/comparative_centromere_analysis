"""Complexity vs. independent functional state on real T2T-CHM13 chr21.

Tests the proposal's central biological question: do the complexity metrics
(Shannon H, Entropy-Rank Ratio R, compression, and real Evo 2 surprise) predict
INDEPENDENT functional state — Fiber-seq chromatin accessibility and CenSat
centromere annotation — beyond GC content?

Inputs:  results/tables/map_windows_functional.csv
Outputs: results/tables/functional_correlations.csv
         results/tables/functional_nested_models.csv
         results/tables/censat_class_means.csv
         results/figures/functional_correlation.png
         results/figures/figS07_censat_class_metrics.png
         results/functional_run_log.txt
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lib_plotting import violin_box  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
LOG = []
METRICS = ["gc", "H1", "H3", "H11", "lz_norm", "gzip_bpb", "bz2_bpb",
           "R_shuffle", "evo2_bpb"]


def log(m=""):
    print(m); LOG.append(str(m))


def cv_auroc(X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    a = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(a)), float(np.std(a))


def main():
    df = pd.read_csv(os.path.join(TAB, "map_windows_functional.csv")).dropna(
        subset=["accessibility"])
    log(f"windows: {len(df)}; accessibility mean {df.accessibility.mean():.2f}, "
        f"median {df.accessibility.median():.2f}")

    # ---- 1. Spearman correlation of each metric with accessibility ----
    log("\n== Spearman correlation with Fiber-seq accessibility ==")
    rows = []
    for m in METRICS:
        rho, p = spearmanr(df[m], df.accessibility)
        rows.append({"metric": m, "spearman_rho": rho, "p_value": p})
        log(f"  {m:10s} rho = {rho:+.3f}  (p = {p:.2e})")
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "functional_correlations.csv"),
                             index=False)

    # ---- 2. Does complexity predict HIGH accessibility beyond GC? ----
    # Binary target: top-quartile accessibility (functionally active windows).
    thr = df.accessibility.quantile(0.75)
    y = (df.accessibility >= thr).astype(int).values
    log(f"\n== Predict top-quartile accessibility (thr={thr:.2f}, "
        f"{y.sum()}/{len(y)} positives) ==")
    models = {
        "GC only": ["gc"],
        "GC + H1": ["gc", "H1"],
        "GC + H1 + R_shuffle": ["gc", "H1", "R_shuffle"],
        "GC + H1 + evo2": ["gc", "H1", "evo2_bpb"],
        "GC + H1 + R + evo2": ["gc", "H1", "R_shuffle", "evo2_bpb"],
        "all complexity": METRICS,
    }
    mrows = []
    for name, fs in models.items():
        m, s = cv_auroc(df[fs].values, y)
        mrows.append({"model": name, "auroc": m, "sd": s})
        log(f"  {name:22s} AUROC = {m:.3f} ± {s:.3f}")
    for f in ["evo2_bpb", "R_shuffle", "bz2_bpb", "H1", "gc"]:
        a = roc_auc_score(y, df[f].values); a = max(a, 1 - a)
        mrows.append({"model": f"[single] {f}", "auroc": a, "sd": 0.0})
        log(f"     [single] {f:10s} AUROC = {a:.3f}")
    pd.DataFrame(mrows).to_csv(os.path.join(TAB, "functional_nested_models.csv"),
                             index=False)

    # ---- 3. Centromere biology: metrics across CenSat classes ----
    keep = ["non_satellite", "alpha_mono", "alpha_hor", "hsat"]
    sub = df[df.censat_class.isin(keep)]
    means = sub.groupby("censat_class")[
        ["gc", "H1", "H11", "bz2_bpb", "R_shuffle", "evo2_bpb", "accessibility"]
    ].mean().reindex(keep)
    means.to_csv(os.path.join(TAB, "censat_class_means.csv"))
    log("\n== metric means by CenSat class (real annotation) ==")
    log(means.round(3).to_string())

    # alpha_hor (centromere core) vs non_satellite separability
    s2 = sub[sub.censat_class.isin(["alpha_hor", "non_satellite"])]
    y2 = (s2.censat_class == "alpha_hor").astype(int).values
    log("\n== alpha-HOR (centromere core) vs non-satellite: single-feature AUROC ==")
    for f in ["R_shuffle", "evo2_bpb", "bz2_bpb", "H11", "accessibility", "gc"]:
        a = roc_auc_score(y2, s2[f].values); a = max(a, 1 - a)
        log(f"     {f:13s} AUROC = {a:.3f}")

    figures(df, sub, keep, rows)
    with open(os.path.join(ROOT, "results", "functional_run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("\nFunctional analysis complete.")


def figures(df, sub, keep, corr_rows):
    # metric-vs-accessibility scatter + regression (small multiples) — shows the
    # actual (weak) relationship distribution, not just a summary correlation.
    cr = pd.DataFrame(corr_rows).set_index("metric").loc[METRICS]
    ncol = 3; nrow = (len(METRICS) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow))
    axes = np.array(axes).flatten()
    for i, m in enumerate(METRICS):
        ax = axes[i]
        d = sub[[m, "accessibility"]].dropna()
        x = d[m].values; y = d["accessibility"].values
        ax.scatter(x, y, s=7, alpha=0.2, color="#1f77b4", edgecolor="none")
        if len(x) > 2 and np.ptp(x) > 0:
            b, a = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr, a + b * xr, color="#d62728", lw=1.8)
        ax.set_title(f"{m}   (ρ = {cr.loc[m, 'spearman_rho']:+.2f})", fontsize=9)
        ax.set_xlabel(m, fontsize=8); ax.set_ylabel("accessibility", fontsize=8)
    for j in range(len(METRICS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Complexity metrics vs Fiber-seq chromatin accessibility (chr21):\n"
                 "scatter + regression — the relationship is weak/absent (no metric predicts accessibility)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "functional_correlation.png"), dpi=130)
    plt.close(fig)

    # censat-class boxplots
    feats = ["evo2_bpb", "R_shuffle", "bz2_bpb", "accessibility"]
    titles = ["Evo 2 AI-surprise", "Entropy-Rank Ratio R", "bzip2 bits/base",
              "Fiber-seq accessibility"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, f, t in zip(axes, feats, titles):
        violin_box(ax, [sub[sub.censat_class == c][f].dropna().values for c in keep],
                   keep, ["#c6c1e0"] * len(keep))
        ax.set_title(t, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Complexity & function across CenSat centromere annotation (real T2T-CHM13 chr21)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS07_censat_class_metrics.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
