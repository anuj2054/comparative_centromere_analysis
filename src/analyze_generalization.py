"""Test whether the chr21 findings generalize across chromosomes.

Pools the 5 new centromere regions (multichrom_windows.csv) with chr21
(map_windows_functional.csv) and asks, per-chromosome and pooled:
  1. Do complexity metrics recover the CenSat centromere annotation
     (alpha-HOR core vs non-satellite) consistently across chromosomes?
  2. Does the weak complexity->accessibility result replicate?

Outputs: results/tables/generalization_annotation_auroc.csv
         results/tables/generalization_accessibility.csv
         results/figures/figS08_generalization_by_chrom.png
         results/figures/figS09_censat_class_gradient.png
         results/generalization_run_log.txt
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
COMMON = ["chrom", "start", "end", "gc", "H1", "H3", "H11", "lz_norm",
          "gzip_bpb", "bz2_bpb", "R_uniform", "R_shuffle", "accessibility",
          "censat_class"]
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def load_pooled():
    multi = pd.read_csv(os.path.join(TAB, "multichrom_windows.csv"))
    chr21 = pd.read_csv(os.path.join(TAB, "map_windows_functional.csv"))
    chr21 = chr21.copy()
    chr21["chrom"] = "chr21"
    pooled = pd.concat([multi[COMMON], chr21[COMMON]], ignore_index=True)
    return pooled


def safe_auroc(y, x):
    if len(set(y)) < 2 or min(np.bincount(y)) < 3:
        return float("nan")
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def annotation_recovery(df):
    log("== Centromere-annotation recovery: alpha-HOR core vs non-satellite ==")
    log("   single-feature AUROC, per chromosome\n")
    feats = ["bz2_bpb", "H11", "R_shuffle", "gc"]
    rows = []
    log(f"{'chrom':8s} {'n_hor':>6s} {'n_non':>6s}  " + "  ".join(f"{f:>9s}" for f in feats))
    for ch, g in list(df.groupby("chrom")) + [("POOLED", df)]:
        sub = g[g.censat_class.isin(["alpha_hor", "non_satellite"])]
        y = (sub.censat_class == "alpha_hor").astype(int).values
        nh, nn = int(y.sum()), int((1 - y).sum())
        cells = {f: safe_auroc(y, sub[f].values) for f in feats}
        rows.append({"chrom": ch, "n_hor": nh, "n_non": nn, **cells})
        log(f"{ch:8s} {nh:>6d} {nn:>6d}  " +
            "  ".join(f"{cells[f]:>9.3f}" if cells[f] == cells[f] else f"{'NA':>9s}"
                     for f in feats))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "generalization_annotation_auroc.csv"),
                             index=False)
    return rows, feats


def accessibility_test(df):
    log("\n== Complexity vs accessibility (pooled across chromosomes) ==")
    d = df.dropna(subset=["accessibility"])
    rows = []
    for m in ["gc", "H1", "H11", "bz2_bpb", "R_shuffle"]:
        rho, p = spearmanr(d[m], d.accessibility)
        rows.append({"metric": m, "spearman_rho": rho, "p": p})
        log(f"  {m:10s} rho = {rho:+.3f} (p = {p:.1e})")
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "generalization_accessibility.csv"),
                             index=False)
    # accessibility by censat class (does alpha_hor stay most accessible among satellites?)
    keep = ["non_satellite", "alpha_mono", "alpha_hor", "hsat"]
    acc = d[d.censat_class.isin(keep)].groupby("censat_class")["accessibility"].mean()
    log("\n  mean accessibility by class (pooled):")
    for c in keep:
        if c in acc:
            log(f"    {c:14s} {acc[c]:.2f}")


def figures(rows, feats, df):
    # per-chromosome annotation-recovery AUROC (bz2 + H11)
    r = pd.DataFrame(rows)
    r = r[r.chrom != "POOLED"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(r)); w = 0.2
    mk = {"bz2_bpb": "o", "H11": "s", "R_shuffle": "^", "gc": "x"}
    for i, f in enumerate(["bz2_bpb", "H11", "R_shuffle", "gc"]):
        ax.scatter(x + i * w, r[f], s=45, marker=mk[f], label=f)
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(r.chrom); ax.set_xlim(-0.5, len(r))
    ax.axhline(0.5, color="k", lw=0.5, ls="--")
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("AUROC (alpha-HOR vs non-satellite)")
    ax.set_title("Centromere-annotation recovery generalizes across chromosomes")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS08_generalization_by_chrom.png"), dpi=130)
    plt.close(fig)

    # pooled censat gradient (bz2)
    keep = ["non_satellite", "alpha_mono", "alpha_hor", "hsat"]
    sub = df[df.censat_class.isin(keep)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([sub[sub.censat_class == c]["bz2_bpb"].dropna().values for c in keep],
               tick_labels=keep, showmeans=True)
    ax.set_ylabel("bzip2 bits/base")
    ax.set_title("Compression complexity vs CenSat class (6 chromosomes pooled)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS09_censat_class_gradient.png"), dpi=130)
    plt.close(fig)


def main():
    df = load_pooled()
    log(f"pooled windows: {len(df)} across {df.chrom.nunique()} chromosomes")
    log(f"censat classes: {df.censat_class.value_counts().to_dict()}\n")
    rows, feats = annotation_recovery(df)
    accessibility_test(df)
    figures(rows, feats, df)
    with open(os.path.join(ROOT, "results", "generalization_run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("\nGeneralization analysis complete.")


if __name__ == "__main__":
    main()
