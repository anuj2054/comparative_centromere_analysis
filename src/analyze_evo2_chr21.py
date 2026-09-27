"""Integrate the REAL Evo 2 AI-surprise (computed on a g5.2xlarge A10G GPU,
model evo2_7b, bf16) into the complexity benchmark.

Inputs:
  results/tables/class_features.csv   (local complexity metrics, by label+frag)
  results/evo2/evo2_class.csv         (Evo 2 bits/base, by label+frag)
  results/tables/map_windows.csv      (local metrics across chr21, by start)
  results/evo2/evo2_map.csv           (Evo 2 bits/base across chr21, by start)

Outputs (all real):
  results/tables/class_features_evo2.csv
  results/tables/evo2_class_summary.csv
  results/tables/evo2_nested_models.csv
  results/figures/evo2_class_box.png
  results/figures/evo2_vs_metrics.png
  results/figures/figS06_chr21_map_evo2.png
  results/evo2_run_log.txt
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from lib_plotting import violin_box  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
EVO = os.path.join(ROOT, "results", "evo2")
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def merge_class():
    feats = pd.read_csv(os.path.join(TAB, "class_features.csv"))
    evo = pd.read_csv(os.path.join(EVO, "evo2_class.csv"))
    df = feats.merge(evo[["label", "frag", "evo2_bpb"]], on=["label", "frag"], how="inner")
    df.to_csv(os.path.join(TAB, "class_features_evo2.csv"), index=False)
    log(f"merged class table: {len(df)} fragments "
        f"({df['label'].value_counts().to_dict()})")
    return df


def cv_auroc(X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    a = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(a)), float(np.std(a))


def contrast(df, pos, neg, seed=0):
    log(f"\n== {pos} vs {neg} (with Evo 2) ==")
    sub = df[df.label.isin([pos, neg])]
    y = (sub.label == pos).astype(int).values
    models = {
        "GC + H1": ["gc", "H1"],
        "GC + H1 + R_shuffle": ["gc", "H1", "R_shuffle"],
        "GC + H1 + evo2": ["gc", "H1", "evo2_bpb"],
        "GC + H1 + R + evo2": ["gc", "H1", "R_shuffle", "evo2_bpb"],
        "all + evo2": ["gc", "H1", "H3", "H11", "lz_norm", "gzip_bpb",
                       "bz2_bpb", "R_shuffle", "evo2_bpb"],
    }
    rows = []
    for name, fs in models.items():
        m, s = cv_auroc(sub[fs].values, y, seed)
        rows.append({"contrast": f"{pos}_vs_{neg}", "model": name,
                     "auroc": m, "sd": s})
        log(f"  {name:24s} AUROC = {m:.3f} ± {s:.3f}")
    # single features
    for f in ["evo2_bpb", "R_shuffle", "bz2_bpb", "lz_norm", "H1"]:
        a = roc_auc_score(y, sub[f].values); a = max(a, 1 - a)
        rows.append({"contrast": f"{pos}_vs_{neg}", "model": f"[single] {f}",
                     "auroc": a, "sd": 0.0})
        log(f"     [single] {f:10s} AUROC = {a:.3f}")
    return rows


def figures(df):
    order = ["coding", "unique", "satellite"]
    # violin + box of Evo2 surprise
    fig, ax = plt.subplots(figsize=(5, 4))
    violin_box(ax, [df[df.label == o]["evo2_bpb"].values for o in order],
               order, ["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylabel("Evo 2 AI-surprise (bits/base)")
    ax.set_title("Evo 2 next-token surprise by class\n(evo2_7b, real GPU inference)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "evo2_class_box.png"), dpi=130)
    plt.close(fig)

    # evo2 vs complexity metrics scatter
    colors = {"coding": "#1f77b4", "unique": "#2ca02c", "satellite": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for lab, g in df.groupby("label"):
        axes[0].scatter(g["bz2_bpb"], g["evo2_bpb"], s=14, alpha=0.6,
                        color=colors[lab], label=lab)
        axes[1].scatter(g["R_shuffle"], g["evo2_bpb"], s=14, alpha=0.6,
                        color=colors[lab], label=lab)
    axes[0].set_xlabel("bzip2 bits/base (compression complexity)")
    axes[0].set_ylabel("Evo 2 AI-surprise (bits/base)")
    axes[1].set_xlabel("Entropy-Rank Ratio R (shuffle null)")
    axes[1].set_ylabel("Evo 2 AI-surprise (bits/base)")
    axes[0].set_title("Evo 2 surprise vs compression")
    axes[1].set_title("Evo 2 surprise vs R")
    axes[0].legend()
    fig.suptitle("Evo 2 captures a different axis: coding is compressible-but-complex "
                 "yet LOW surprise")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "evo2_vs_metrics.png"), dpi=130)
    plt.close(fig)

    # correlations
    for a, b in [("evo2_bpb", "bz2_bpb"), ("evo2_bpb", "R_shuffle"),
                 ("evo2_bpb", "H1")]:
        log(f"corr({a},{b}) = {df[[a, b]].corr().iloc[0,1]:.3f}")


def map_figure():
    mp = pd.read_csv(os.path.join(TAB, "map_windows.csv"))
    ev = pd.read_csv(os.path.join(EVO, "evo2_map.csv"))
    m = mp.merge(ev[["start", "evo2_bpb"]], on="start", how="inner")
    m.to_csv(os.path.join(TAB, "map_windows_evo2.csv"), index=False)
    x = (m.start + m.end) / 2 / 1e6
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    sk = dict(s=5, alpha=0.5, edgecolors="none")
    axes[0].scatter(x, m.H1, color="#555", **sk); axes[0].set_ylabel("H₁")
    axes[0].set_title("chr21 information-topology map with real Evo 2 surprise")
    axes[1].scatter(x, m.bz2_bpb, color="#9467bd", **sk); axes[1].set_ylabel("bz2 bpb")
    axes[2].scatter(x, m.R_shuffle, color="#d62728", **sk); axes[2].set_ylabel("R (shuffle)")
    axes[3].scatter(x, m.evo2_bpb, color="#1f77b4", **sk); axes[3].set_ylabel("Evo2 bpb")
    axes[3].set_xlabel("chr21 position (Mb)")
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS06_chr21_map_evo2.png"), dpi=130)
    plt.close(fig)
    log(f"\nmap merged: {len(m)} windows; "
        f"corr(evo2, R_shuffle)={m[['evo2_bpb','R_shuffle']].corr().iloc[0,1]:.3f}; "
        f"corr(evo2, bz2)={m[['evo2_bpb','bz2_bpb']].corr().iloc[0,1]:.3f}")


def main():
    df = merge_class()
    summ = df.groupby("label")["evo2_bpb"].agg(["mean", "std", "count"])
    summ.to_csv(os.path.join(TAB, "evo2_class_summary.csv"))
    log("\n== Evo 2 AI-surprise by class (bits/base) ==")
    log(summ.round(4).to_string())

    rows = []
    rows += contrast(df, "satellite", "unique")
    rows += contrast(df, "satellite", "coding", seed=1)
    rows += contrast(df, "coding", "unique", seed=2)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "evo2_nested_models.csv"), index=False)

    figures(df)
    map_figure()

    with open(os.path.join(ROOT, "results", "evo2_run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("\nIntegration complete.")


if __name__ == "__main__":
    sys.exit(main())
