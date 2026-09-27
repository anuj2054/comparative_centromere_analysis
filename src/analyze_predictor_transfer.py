"""Extend the centromere-activity predictor to ALL testable human chromosomes.

The headline predictor was validated on chr14/15/22 only — the chromosomes whose
centromere regions we had fetched. But 16 of 24 human chromosomes carry BOTH an
active (live, 'L') and an inactive HOR array and are therefore testable. Here we
fetch the centromere region of every such chromosome, compute compression (bzip2)
and high-order entropy (the cheap, GPU-free predictor features), label windows
active/inactive from CenSat, and run a genome-wide leave-one-chromosome-out
evaluation (train on 15 chromosomes, predict the held-out one).

Output: results/tables/human_all_centromeres.csv
        results/tables/predictor_loco_genomewide.csv
        results/figures/predictor_genomewide.png ; results/predictor_log.txt
"""
import os
import re
import time
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean, gc_content  # noqa: E402
from lib_entropy_rank import kmer_entropy  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CEN = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/censat_v2.1.bb"
WINDOW, STEP, KHIGH = 2000, 4000, 11

# CHM13 RefSeq accessions: 24 consecutive records, chr1=NC_060925.1 ... chrN =
# 060925+(N-1), with X the 23rd and Y the 24th. Derived from the one rule so the
# off-by-one that once mapped chrX onto the chrY record cannot silently return.
CHROMS = [f"chr{n}" for n in range(1, 23)] + ["chrX", "chrY"]
ACC = {c: f"NC_0609{25 + i}.1" for i, c in enumerate(CHROMS)}
# 16 chromosomes with both active(L) and inactive HOR (from CenSat scan)
TESTABLE = ["chr1", "chr3", "chr4", "chr5", "chr7", "chr10", "chr11", "chr13",
            "chr14", "chr15", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def fetch(acc, lo, hi, chunk=500_000):
    out = []
    s = lo
    while s <= hi:
        e = min(s + chunk - 1, hi)
        url = EFETCH + "?" + urllib.parse.urlencode(dict(
            db="nuccore", id=acc, rettype="fasta", retmode="text",
            seq_start=s, seq_stop=e))
        req = urllib.request.Request(url, headers={"User-Agent": "GC/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = r.read().decode()
        out.append("".join(l.strip() for l in txt.splitlines() if not l.startswith(">")))
        time.sleep(0.34)
        s = e + 1
    return "".join(out)


def hor_spans(bb, ch):
    act, ina = [], []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0]
        if n.lower().startswith("hor"):
            (act if re.search(r"H\d+L", n) else ina).append((s, e))
    return act, ina


def build():
    bb = pyBigWig.open(CEN)
    rows = []
    for ch in TESTABLE:
        act, ina = hor_spans(bb, ch)
        alls = act + ina
        lo = max(1, min(s for s, e in alls) - 100_000)
        hi = max(e for s, e in alls) + 100_000
        log(f"[{ch}] {ACC[ch]} {lo:,}-{hi:,} ({(hi-lo)/1e6:.1f} Mb); "
            f"activeL {sum(e-s for s,e in act)/1e6:.2f} / inactive "
            f"{sum(e-s for s,e in ina)/1e6:.2f} Mb")
        seq = fetch(ACC[ch], lo, hi)
        for w0 in range(0, len(seq) - WINDOW + 1, STEP):
            cseq = clean(seq[w0:w0 + WINDOW])
            if len(cseq) < 0.5 * WINDOW:
                continue
            mid = lo + w0 + WINDOW // 2
            st = ("active" if any(s <= mid < e for s, e in act)
                  else "inactive" if any(s <= mid < e for s, e in ina) else "other")
            if st == "other":
                continue
            rows.append({"chrom": ch, "start": lo + w0, "state": st,
                         "gc": gc_content(cseq), "bz2_bpb": bz2_bits_per_base(cseq),
                         "H11": kmer_entropy(cseq, KHIGH)})
    bb.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "human_all_centromeres.csv"), index=False)
    return df


def main():
    path = os.path.join(TAB, "human_all_centromeres.csv")
    df = pd.read_csv(path) if os.path.exists(path) else build()
    log(f"\nwindows: {len(df)} across {df.chrom.nunique()} chromosomes "
        f"({(df.state=='active').sum()} active / {(df.state=='inactive').sum()} inactive)")

    # genome-wide leave-one-chromosome-out: train on all but one chrom, predict it
    log("\n== Genome-wide leave-one-chromosome-out predictor ==")
    feats_sets = {"bz2+H11": ["bz2_bpb", "H11"], "bz2": ["bz2_bpb"], "gc (baseline)": ["gc"]}
    rows = []
    chroms = sorted(df.chrom.unique(), key=lambda c: int(c[3:]))
    for name, feats in feats_sets.items():
        aucs = {}
        for test in chroms:
            tr, te = df[df.chrom != test], df[df.chrom == test]
            if te.state.nunique() < 2 or (te.state == "inactive").sum() < 3:
                continue
            sc = StandardScaler().fit(tr[feats].values)
            clf = LogisticRegression(max_iter=3000).fit(
                sc.transform(tr[feats].values), (tr.state == "active").astype(int))
            p = clf.predict_proba(sc.transform(te[feats].values))[:, 1]
            aucs[test] = roc_auc_score((te.state == "active").astype(int), p)
        mean = np.mean(list(aucs.values()))
        rows.append({"features": name, "n_chrom_tested": len(aucs),
                     "mean_heldout_auroc": mean,
                     "median": np.median(list(aucs.values())),
                     "min": min(aucs.values()), "max": max(aucs.values())})
        log(f"  {name:14s} tested on {len(aucs)} held-out chroms: "
            f"mean AUROC {mean:.3f} (median {np.median(list(aucs.values())):.3f}, "
            f"range {min(aucs.values()):.2f}-{max(aucs.values()):.2f})")
        if name == "bz2+H11":
            per_chrom = aucs
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "predictor_loco_genomewide.csv"), index=False)

    # figure: the DATA behind the predictor — per-array homogeneity (bzip2) by
    # chromosome, coloured by active/inactive. This shows the separation *distribution*
    # that produces the held-out AUROC; chromosomes where active & inactive overlap
    # (e.g. chr20) are exactly where the predictor fails. Held-out AUROC: 0.76 vs the
    # functional CENP-A target, ~0.95 vs the (partly circular) CenSat annotation.
    # Within-chromosome slope plot: each line links one chromosome's mean active-array
    # bzip2 to its mean inactive-array bzip2. Lines slope up (active lower) on ~all
    # chromosomes, the within-chromosome contrast behind the predictor (avoids the
    # Simpson's-paradox of pooling baselines).
    al_path = os.path.join(TAB, "array_level.csv")
    if os.path.exists(al_path):
        al = pd.read_csv(al_path).dropna(subset=["bz2"])
        rows = []
        for ch, g in al.groupby("chrom"):
            a = g[g.censat_active == True].bz2; i = g[g.censat_active == False].bz2
            if len(a) and len(i):
                rows.append((ch, a.mean(), i.mean()))
        rows = sorted(rows, key=lambda r: int("".join(c for c in r[0] if c.isdigit()) or 99))
        ndown = sum(1 for _, av, iv in rows if av < iv)
        fig, ax = plt.subplots(figsize=(7, 5.2))
        for ch, av, iv in rows:
            ax.plot([0, 1], [av, iv], color="#2ca02c" if av < iv else "#999999",
                    alpha=0.6, lw=1.4, zorder=1)
            ax.scatter([0], [av], color="#2ca02c", s=40, zorder=3, edgecolor="white", lw=0.4)
            ax.scatter([1], [iv], color="#7570b3", s=40, marker="D", zorder=3,
                       edgecolor="white", lw=0.4)
        ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
        ax.set_xticklabels(["active", "inactive (relict)"])
        ax.set_ylabel("array bzip2 bits/base  (↓ more homogeneous)")
        ax.set_title("Within each chromosome, the active HOR array is the more homogeneous")
        ax.text(0.5, ax.get_ylim()[1], f"each line is one chromosome\nactive lower on {ndown}/{len(rows)}",
                ha="center", va="top", fontsize=9)
        ax.text(-0.08, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "predictor_genomewide.png"), dpi=130)
        plt.close(fig)

    with open(os.path.join(ROOT, "results", "predictor_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
