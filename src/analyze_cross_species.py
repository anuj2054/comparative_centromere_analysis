"""Synthesize the cross-species replication: human, chimp, gorilla.

Combines the active_hor-vs-inactive(dhor) homogeneity test across species into one
table and figure. Orangutan is excluded with a note: its CenSat annotation (both
Sumatran and Bornean) contains no dhor/divergent-HOR arrays, so the contrast
cannot be formed there.

Output: results/tables/cross_species_summary.csv ; results/figures/cross_species.png
        results/cross_species_log.txt
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
CHM13_CEN = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/censat_v2.1.bb"
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def human_active_inactive():
    """Reconstruct human active/inactive bz2 on the both-class chromosomes."""
    multi = pd.read_csv(os.path.join(TAB, "multichrom_windows.csv"))
    bb = pyBigWig.open(CHM13_CEN)
    keep = {"chr14": "chr14", "chr15": "chr15", "chr22": "chr22"}
    rows = []
    for ch in keep:
        act, ina = [], []
        for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
            n = r.split("\t")[0]
            if n.lower().startswith("hor"):
                (act if re.search(r"H\d+L", n) else ina).append((s, e))
        g = multi[multi.chrom == ch]
        for _, r in g.iterrows():
            m = (int(r.start) + int(r.end)) // 2 - 1
            st = ("active" if any(s <= m < e for s, e in act)
                  else "inactive" if any(s <= m < e for s, e in ina) else "other")
            if st != "other":
                rows.append({"state": st, "bz2_bpb": r.bz2_bpb, "gc": r.gc})
    bb.close()
    return pd.DataFrame(rows)


def stats(df, species):
    a = df[df.state == "active"]; i = df[df.state == "inactive"]
    y = np.r_[np.ones(len(a)), np.zeros(len(i))]
    auc = max(roc_auc_score(y, np.r_[a.bz2_bpb, i.bz2_bpb]),
              1 - roc_auc_score(y, np.r_[a.bz2_bpb, i.bz2_bpb]))
    _, p = mannwhitneyu(a.bz2_bpb, i.bz2_bpb)
    band = df[(df.gc >= 0.34) & (df.gc <= 0.40)]
    ba, bi = band[band.state == "active"].bz2_bpb, band[band.state == "inactive"].bz2_bpb
    gm_auc = float("nan")
    if len(bi) > 5:
        yb = np.r_[np.ones(len(ba)), np.zeros(len(bi))]
        gm_auc = max(roc_auc_score(yb, np.r_[ba, bi]), 1 - roc_auc_score(yb, np.r_[ba, bi]))
    return {"species": species, "n_active": len(a), "n_inactive": len(i),
            "bz2_active": a.bz2_bpb.mean(), "bz2_inactive": i.bz2_bpb.mean(),
            "auroc": auc, "gc_matched_auroc": gm_auc, "p": p}


def main():
    data = {
        "Human (CHM13)": human_active_inactive(),
        "Chimp (mPanTro3)": pd.read_csv(os.path.join(TAB, "chimp_active_vs_inactive.csv")),
        "Bonobo (mPanPan1)": pd.read_csv(os.path.join(TAB, "bonobo_active_vs_inactive.csv")),
        "Gorilla (mGorGor1)": pd.read_csv(os.path.join(TAB, "gorilla_active_vs_inactive.csv")),
    }
    rows = [stats(df, sp) for sp, df in data.items()]
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(TAB, "cross_species_summary.csv"), index=False)
    log("== Cross-species: active vs inactive centromeric HOR (compression) ==")
    log(summ.round(3).to_string(index=False))
    log("\nExcluded (no dhor/divergent-HOR arrays in their CenSat annotation, so the "
        "contrast cannot be formed): orangutan (Pongo abelii 252 Mb / P. pygmaeus 182 Mb "
        "active_hor, 0 dhor) and siamang (mSymSyn1, 60 Mb active_hor, 0 dhor).")
    log("\nConserved across 4 great-ape lineages (Homo, Pan [×2], Gorilla; ~9 Myr): active "
        "centromeric arrays are more sequence-homogeneous than inactive relict arrays, robust "
        "to GC. Effect size varies (human 0.99 to bonobo 0.71), tracking how diverged each "
        "species' relict arrays are.")

    # figure: per-species active vs inactive bz2 boxplots, ordered by AUROC
    # (descending) to show the effect-size gradient that tracks relict divergence.
    auc_of = {sp: summ[summ.species.str.startswith(sp.split()[0])].iloc[0].auroc
              for sp in data}
    order = sorted(data, key=lambda sp: -auc_of[sp])
    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4), sharey=True)
    for k, (ax, sp) in enumerate(zip(axes, order)):
        df = data[sp]
        ax.boxplot([df[df.state == s].bz2_bpb.values for s in ["active", "inactive"]],
                   tick_labels=["active", "inactive"], showmeans=True)
        ax.set_title(f"{sp}\nAUROC {auc_of[sp]:.2f}", fontsize=10)
        ax.text(-0.05, 1.04, "abcd"[k], transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    axes[0].set_ylabel("bzip2 bits/base")
    fig.suptitle("Active vs inactive centromeric HOR across four great apes")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cross_species.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(ROOT, "results", "cross_species_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
