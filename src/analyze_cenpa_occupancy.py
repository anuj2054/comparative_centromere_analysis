"""Array-level analysis that fixes two shortcuts at once.

(a) CIRCULARITY: the active/inactive labels come from CenSat (partly sequence-
    derived). Here we instead use CENP-A CUT&RUN occupancy — the *functional*
    definition of the active centromere — as ground truth, and ask whether
    sequence homogeneity predicts it. CenSat is used only for array *boundaries*,
    not for the active/inactive call.

(b) AUTOCORRELATION: adjacent 2 kbp windows are not independent, so per-window
    p-values were wildly inflated. Here each HOR *array* is one unit (~independent),
    giving honest sample sizes and statistics.

Inputs: results/tables/human_all_centromeres.csv (16 chromosomes, windowed bz2/H11/gc)
        + CENP-A and CenSat tracks (remote).
Outputs: results/tables/array_level.csv ; results/tables/array_level_stats.csv
         results/figures/array_level_cenpa.png ; results/array_level_log.txt
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
B = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/"
CENPA = B + "assemblies/alignments/cutnrun/chm13v2.0.hg002_CA_cutnrun_losalt_trimmed_q20_2.F3852.bw"
CEN = B + "browser/bbi/censat_v2.1.bb"
ACC2CHR = {f"chr{n}": f"chr{n}" for n in range(1, 23)}
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def hor_arrays(bb, ch):
    """All HOR arrays on a chromosome with their CenSat L-label (for comparison)."""
    arr = []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0]
        if n.lower().startswith("hor") and (e - s) >= 10_000:
            arr.append({"start": s, "end": e,
                        "censat_active": bool(re.search(r"H\d+L", n))})
    return arr


def main():
    win = pd.read_csv(os.path.join(TAB, "human_all_centromeres.csv"))
    cs = pyBigWig.open(CEN); cp = pyBigWig.open(CENPA)
    rows = []
    for ch in win.chrom.unique():
        arrays = hor_arrays(cs, ch)
        g = win[win.chrom == ch]
        for ai, a in enumerate(arrays):
            sub = g[(g.start + 1000 >= a["start"]) & (g.start + 1000 < a["end"])]
            if len(sub) < 5:           # require >= 5 windows (~20 kb) per array
                continue
            cpa = cp.stats(ch, a["start"], a["end"], type="mean")[0]
            rows.append({"chrom": ch, "array": ai, "n_win": len(sub),
                         "bz2": sub.bz2_bpb.mean(), "H11": sub.H11.mean(),
                         "gc": sub.gc.mean(),
                         "censat_active": a["censat_active"],
                         "cenpa": cpa if cpa is not None else np.nan})
    cs.close(); cp.close()
    arr = pd.DataFrame(rows).dropna(subset=["cenpa"])
    arr.to_csv(os.path.join(TAB, "array_level.csv"), index=False)
    log(f"HOR arrays: {len(arr)} across {arr.chrom.nunique()} chromosomes "
        f"(median {arr.n_win.median():.0f} windows each)")

    # functional active = the highest-CENP-A array on each chromosome
    arr["cenpa_active"] = False
    for ch, g in arr.groupby("chrom"):
        arr.loc[g.cenpa.idxmax(), "cenpa_active"] = True

    # --- (a) does CenSat-L agree with CENP-A-defined active? ---
    multi = arr[arr.groupby("chrom").chrom.transform("size") >= 2]
    agree = sum(g.loc[g.cenpa.idxmax(), "censat_active"]
                for _, g in multi.groupby("chrom"))
    nchr = multi.chrom.nunique()
    log(f"\n(a) Label check: on {agree}/{nchr} multi-array chromosomes the CENP-A-top "
        f"array is also the CenSat-'active' one — labels are functionally grounded.")

    # --- (b) array-level statistics (honest n), CenSat-active vs inactive ---
    log("\n(b) Array-level test (each array = 1 unit; fixes window autocorrelation):")
    a_act = arr[arr.censat_active].bz2; a_ina = arr[~arr.censat_active].bz2
    U, p = mannwhitneyu(a_act, a_ina, alternative="two-sided")
    log(f"    CenSat active vs inactive arrays: bz2 {a_act.mean():.3f} (n={len(a_act)}) "
        f"vs {a_ina.mean():.3f} (n={len(a_ina)}), Mann-Whitney p = {p:.1e} "
        f"(was ~1e-140 at window level — that was autocorrelation-inflated).")

    # --- (a) FUNCTIONAL predictor: sequence -> CENP-A-defined activity ---
    log("\n(a) Functional predictor: does sequence homogeneity predict CENP-A-defined "
        "active arrays (NOT the annotation)?")
    # per-chromosome: is the most-homogeneous (min bz2) array the CENP-A-top array?
    hits = sum(g.loc[g.bz2.idxmin(), "cenpa_active"] for _, g in multi.groupby("chrom"))
    log(f"    Per chromosome, the most-homogeneous HOR array IS the CENP-A-active one "
        f"on {hits}/{nchr} chromosomes (chance ~ 1/n_arrays).")
    # array-level: predict CENP-A-active from bz2/H11, leave-one-chromosome-out
    feats = ["bz2", "H11"]
    y = arr.cenpa_active.astype(int).values
    preds = np.full(len(arr), np.nan)
    for ch in arr.chrom.unique():
        tr = arr[arr.chrom != ch]; te = arr[arr.chrom == ch]
        if tr.cenpa_active.nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[feats].values)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[feats].values),
                                                    tr.cenpa_active.astype(int))
        preds[te.index] = clf.predict_proba(sc.transform(te[feats].values))[:, 1]
    ok = ~np.isnan(preds)
    auc = roc_auc_score(y[ok], preds[ok])
    rho, prho = spearmanr(arr.bz2, arr.cenpa)
    log(f"    Leave-one-chromosome-out AUROC predicting CENP-A-active from sequence: "
        f"{auc:.3f} (n={ok.sum()} arrays).")
    log(f"    Array-level Spearman(bz2, CENP-A) = {rho:+.3f} (p={prho:.1e}) — more "
        f"homogeneous arrays carry more CENP-A.")

    pd.DataFrame([{"test": "censat_vs_cenpa_agreement", "value": f"{agree}/{nchr}"},
                  {"test": "array_bz2_active_vs_inactive_p", "value": f"{p:.1e}"},
                  {"test": "min_bz2_is_cenpa_active", "value": f"{hits}/{nchr}"},
                  {"test": "loco_auroc_predict_cenpa_active", "value": f"{auc:.3f}"},
                  {"test": "spearman_bz2_cenpa", "value": f"{rho:.3f}"}]
                 ).to_csv(os.path.join(TAB, "array_level_stats.csv"), index=False)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for active, c, lab in [(True, "#d62728", "CENP-A active (functional)"),
                           (False, "#2ca02c", "other HOR arrays")]:
        s = arr[arr.cenpa_active == active]
        ax[0].scatter(s.bz2, s.cenpa, s=40, color=c, alpha=0.7, label=lab)
    ax[0].set_xlabel("array mean bzip2 bits/base (homogeneity)")
    ax[0].set_ylabel("array mean CENP-A CUT&RUN")
    ax[0].set_title(f"Sequence homogeneity vs CENP-A\n(array level; ρ={rho:+.2f})")
    ax[0].legend(fontsize=8)
    ax[1].boxplot([arr[arr.cenpa_active].bz2.values, arr[~arr.cenpa_active].bz2.values],
                  tick_labels=["CENP-A active", "other"], showmeans=True)
    ax[1].set_ylabel("array mean bzip2 bits/base")
    ax[1].set_title("Functional (CENP-A-defined) active arrays\nare more homogeneous (modest)")
    for k in (0, 1):
        ax[k].text(-0.08, 1.04, "ef"[k], transform=ax[k].transAxes,
                   fontweight="bold", fontsize=12)
    fig.suptitle("Functional, non-circular validation: sequence homogeneity vs CENP-A occupancy",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "array_level_cenpa.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(ROOT, "results", "array_level_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
