"""Cross-species replication in chimpanzee (Pan troglodytes, T2T mPanTro3).

Tests whether the human finding — active centromeric HOR arrays are more
sequence-homogeneous than inactive ones — replicates in chimp. Chimp CenSat
(mPanTro3 v1.2) labels arrays explicitly as `active_hor` vs `dhor` (divergent/
relict HOR), an independent annotation in a species ~6 Myr diverged.

Sequence is pulled region-by-region from the bgzipped chimp assembly via remote
htslib range requests (pysam) — no genome download. Metrics: compression (bzip2)
and high-order Shannon entropy H11, the measures shown to carry the signal in
human (Evo 2 not needed/run here — CPU only).

Output: results/tables/chimp_active_vs_inactive.csv ; results/chimp_log.txt
        results/figures/figS10_chimp_replication.png
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pysam
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean, gc_content  # noqa: E402
from lib_entropy_rank import kmer_entropy  # noqa: E402
from lib_plotting import violin_box  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
CEN = "https://genomeark.s3.amazonaws.com/species/Pan_troglodytes/mPanTro3/assembly_curated/repeats/mPanTro3_v2.0_CenSat_v1.2.bb"
FASTA = "https://genomeark.s3.amazonaws.com/species/Pan_troglodytes/mPanTro3/assembly_curated/intermediates/mPanTro3.hap1.cur.20231031.fasta.gz"
CHROMS = ["chr11_hap1_hsa9", "chr21_hap1_hsa20", "chr2_hap1_hsa3",
          "chr3_hap1_hsa4", "chr5_hap1_hsa6", "chr7_hap1_hsa8"]
WINDOW, STEP, KHIGH = 2000, 4000, 11
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def hor_spans(bb, ch):
    act, ina = [], []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0].lower()
        if n.startswith("active_hor"):
            act.append((s, e))
        elif n.startswith("dhor"):
            ina.append((s, e))
    return act, ina


def label(mid, act, ina):
    for s, e in act:
        if s <= mid < e:
            return "active"
    for s, e in ina:
        if s <= mid < e:
            return "inactive"
    return "other"


def main():
    bb = pyBigWig.open(CEN)
    fa = pysam.FastaFile(FASTA)
    rows = []
    for ch in CHROMS:
        act, ina = hor_spans(bb, ch)
        allspans = act + ina
        lo = max(0, min(s for s, e in allspans) - 100_000)
        hi = max(e for s, e in allspans) + 100_000
        log(f"[{ch}] region {lo:,}-{hi:,} ({(hi-lo)/1e6:.1f} Mb); "
            f"active {sum(e-s for s,e in act)/1e6:.2f} Mb, "
            f"dhor {sum(e-s for s,e in ina)/1e6:.2f} Mb")
        seq = fa.fetch(ch, lo, hi)
        for w0 in range(0, len(seq) - WINDOW + 1, STEP):
            cseq = clean(seq[w0:w0 + WINDOW])
            if len(cseq) < 0.5 * WINDOW:
                continue
            mid = lo + w0 + WINDOW // 2
            st = label(mid, act, ina)
            if st == "other":
                continue
            rows.append({"chrom": ch, "start": lo + w0, "state": st,
                         "gc": gc_content(cseq), "bz2_bpb": bz2_bits_per_base(cseq),
                         "H11": kmer_entropy(cseq, KHIGH)})
    bb.close(); fa.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "chimp_active_vs_inactive.csv"), index=False)

    log(f"\nwindows: {len(df)}  ({df.state.value_counts().to_dict()})")
    log("\n== Chimp: active (active_hor) vs inactive (dhor) ==")
    a = df[df.state == "active"]; i = df[df.state == "inactive"]
    for m in ["bz2_bpb", "H11", "gc"]:
        U, p = mannwhitneyu(a[m], i[m], alternative="two-sided")
        y = np.r_[np.ones(len(a)), np.zeros(len(i))]; x = np.r_[a[m].values, i[m].values]
        auc = max(roc_auc_score(y, x), 1 - roc_auc_score(y, x))
        log(f"  {m:8s} active {a[m].mean():7.3f}  inactive {i[m].mean():7.3f}  "
            f"p={p:.1e}  AUROC={auc:.3f}")

    log("\n== Per-chromosome bz2 (active vs inactive) ==")
    for ch, g in df.groupby("chrom"):
        ga = g[g.state == "active"].bz2_bpb; gi = g[g.state == "inactive"].bz2_bpb
        if len(gi) >= 5:
            log(f"  {ch:20s} active {ga.mean():.3f} (n={len(ga)})  "
                f"inactive {gi.mean():.3f} (n={len(gi)})  delta={ga.mean()-gi.mean():+.3f}")

    log("\n== GC-confound control (5-fold CV AUROC) ==")
    y = (df.state == "active").astype(int).values
    for feats in [["gc"], ["bz2_bpb"], ["gc", "bz2_bpb"], ["gc", "H11"]]:
        X = StandardScaler().fit_transform(df[feats].values)
        auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5,
                              scoring="roc_auc").mean()
        log(f"  {'+'.join(feats):14s} AUROC = {max(auc,1-auc):.3f}")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, m, t in zip(axes, ["bz2_bpb", "H11"],
                        ["bzip2 bits/base", "Shannon H11"]):
        violin_box(ax, [df[df.state == s][m].values for s in ["active", "inactive"]],
                   ["active_hor", "dhor"], ["#0072b2", "#e69f00"])
        ax.set_title(t, fontsize=10)
    fig.suptitle("Chimpanzee (mPanTro3): active vs divergent centromeric HOR — replication")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS10_chimp_replication.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(ROOT, "results", "chimp_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("\nDone.")


if __name__ == "__main__":
    main()
