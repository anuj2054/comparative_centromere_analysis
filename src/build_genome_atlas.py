"""Genome-wide complexity atlas of T2T-CHM13 v2.0 — the whole-genome deliverable.

Computes complexity metrics in non-overlapping windows across ALL 24 chromosomes
(GC, Shannon H1/H3, gzip & bzip2 bits/base), assigns each window a CenSat class
(satellite family or non-satellite), and summarizes how sequence complexity
distributes across the genome and across the repetitive "dark matter".

Evo 2 and the shuffle-null R are NOT run genome-wide (cost/speed); this is the
CPU atlas. Input: a local chm13v2.0.fa.gz + censat BED (see ATLAS_DIR).

Outputs (committed): results/tables/atlas_by_chrom.csv, atlas_by_class.csv,
  results/figures/fig3_genome_atlas.png ; full per-window table -> ATLAS_DIR (gitignored).
"""
import bz2 as _bz2
import gzip
import math
import os
from bisect import bisect_right
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
WORK = os.environ.get("GC_WORK", os.path.join(ROOT, "work"))
# build_atlas_repeats.py and analyze_other_compartments.py read the atlas back from
# here, so the default must survive a reboot. Override with GC_WORK or GC_ATLAS_DIR.
ATLAS_DIR = os.environ.get("GC_ATLAS_DIR", os.path.join(WORK, "gc_atlas"))
FASTA = os.path.join(ATLAS_DIR, "chm13v2.0.fa.gz")
CENSAT = os.path.join(ATLAS_DIR, "censat.bed")
WIN = 10000
MAIN = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
ACGT = set("ACGT")


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else float("nan")


def shannon(s, k):
    if len(s) < k:
        return float("nan")
    c = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    tot = sum(c.values())
    return -sum((v / tot) * math.log2(v / tot) for v in c.values())


def bpb(s, comp):
    return len(comp(s.encode())) * 8 / len(s)


def collapse(name):
    n = name.lower()
    if n.startswith("ct"):
        return "non_satellite"
    if n.startswith("hor"):
        return "alpha_hor"
    if n.startswith("mon"):
        return "alpha_mono"
    if n.startswith("hsat"):
        return "hsat"
    if n.startswith("gsat"):
        return "gsat"
    if n.startswith("bsat"):
        return "bsat"
    if "sst1" in n or n.startswith("censat"):
        return "other_sat"
    return "non_satellite"


def load_censat():
    by = {}
    for line in open(CENSAT):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        by.setdefault(f[0], []).append((int(f[1]), int(f[2]), collapse(f[3])))
    for c in by:
        by[c].sort()
    return by


def classify(intervals, pos):
    """Class of the interval covering pos (binary search on sorted starts)."""
    if not intervals:
        return "non_satellite"
    starts = [s for s, e, c in intervals]
    i = bisect_right(starts, pos) - 1
    if i >= 0 and intervals[i][0] <= pos < intervals[i][1]:
        return intervals[i][2]
    return "non_satellite"


def iter_chroms(path):
    """Yield (name, sequence) from a gzipped FASTA, one chromosome at a time."""
    name, buf = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    yield name, "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name:
        yield name, "".join(buf)


def main():
    censat = load_censat()
    rows = []
    n_chrom = 0
    for name, seq in iter_chroms(FASTA):
        if name not in MAIN:
            continue
        n_chrom += 1
        seq = seq.upper()
        ints = censat.get(name, [])
        print(f"[{name}] {len(seq)/1e6:.1f} Mb ...", flush=True)
        for w0 in range(0, len(seq) - WIN + 1, WIN):
            w = seq[w0:w0 + WIN]
            cs = "".join(ch for ch in w if ch in ACGT)
            if len(cs) < 0.5 * WIN:
                continue
            rows.append((name, w0, gc(cs), shannon(cs, 1), shannon(cs, 3),
                         bpb(cs, lambda b: gzip.compress(b, 6)),
                         bpb(cs, lambda b: _bz2.compress(b, 6)),
                         classify(ints, w0 + WIN // 2)))
    df = pd.DataFrame(rows, columns=["chrom", "start", "gc", "H1", "H3",
                                     "gzip_bpb", "bz2_bpb", "class"])
    os.makedirs(ATLAS_DIR, exist_ok=True)
    df.to_csv(os.path.join(ATLAS_DIR, "genome_atlas_full.csv.gz"), index=False,
              compression="gzip")
    print(f"\nTotal: {len(df):,} windows of {WIN} bp across {n_chrom} chromosomes "
          f"({df['start'].max() and len(df)*WIN/1e9:.2f} Gb covered)")

    by_chrom = df.groupby("chrom")[["gc", "H1", "H3", "gzip_bpb", "bz2_bpb"]].mean()
    by_chrom["n_windows"] = df.groupby("chrom").size()
    by_chrom = by_chrom.reindex([c for c in MAIN if c in by_chrom.index])
    by_chrom.to_csv(os.path.join(TAB, "atlas_by_chrom.csv"))

    by_class = df.groupby("class").agg(
        n_windows=("gc", "size"), frac_genome=("gc", lambda x: len(x) / len(df)),
        gc=("gc", "mean"), H3=("H3", "mean"),
        gzip_bpb=("gzip_bpb", "mean"), bz2_bpb=("bz2_bpb", "mean"))
    by_class = by_class.sort_values("n_windows", ascending=False)
    by_class.to_csv(os.path.join(TAB, "atlas_by_class.csv"))
    print("\n== genome-wide complexity by class ==")
    print(by_class.round(3).to_string())

    # low-complexity fraction of the genome
    lowc = (df.bz2_bpb < 1.9).mean()
    print(f"\nfraction of genome with bz2 < 1.9 bits/base (compressible/repetitive): "
          f"{100*lowc:.1f}%")

    figures(df, by_class)
    print("\nAtlas complete.")


def violin_box(ax, data, labels, colors, widths=0.78):
    """Violin (translucent, coloured) with a narrow dark box-and-whisker inside."""
    pos = list(range(1, len(data) + 1))
    vp = ax.violinplot(data, positions=pos, showextrema=False, widths=widths)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.45); body.set_edgecolor(c); body.set_linewidth(0.9)
    ax.boxplot(data, positions=pos, widths=0.11, showfliers=False, showmeans=True,
               patch_artist=True, medianprops=dict(color="white", lw=1.3),
               meanprops=dict(marker="o", mfc="white", mec="#222222", ms=3.5, mew=0.7),
               boxprops=dict(facecolor="#3a3a3a", edgecolor="#3a3a3a"),
               whiskerprops=dict(color="#3a3a3a", lw=1.0), capprops=dict(color="#3a3a3a", lw=1.0))
    ax.set_xticks(pos); ax.set_xticklabels(labels)


def figures(df, by_class):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # (a) genome-wide compression distribution, satellite vs non-satellite
    sat = df[df["class"] != "non_satellite"]["bz2_bpb"].dropna().values
    non = df[df["class"] == "non_satellite"]["bz2_bpb"].dropna().values
    violin_box(axes[0], [non, sat], ["non-satellite", "satellite\n(CenSat)"],
               ["#0072b2", "#e69f00"])   # colourblind-safe blue/orange
    axes[0].set_ylabel("bzip2 bits/base")
    axes[0].set_title("Genome is bimodal: satellite vs non-satellite")
    # (b) compression distribution per satellite class
    order = by_class.index.tolist()
    violin_box(axes[1], [df[df["class"] == c].bz2_bpb.dropna().values for c in order],
               order, ["#c6c1e0"] * len(order))
    axes[1].set_ylabel("bzip2 bits/base (per 10 kb window)"); axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Only tandem satellite is low-complexity")
    # (c) RepeatMasker breakdown of 'non-satellite': per-window bzip2 distribution per
    # interspersed-repeat class (all stay high, far above alpha-HOR).
    rw_path = os.path.join(TAB, "atlas_repeat_windows.csv")
    if os.path.exists(rw_path):
        rw = pd.read_csv(rw_path)
        rcl = [c for c in ["LINE", "SINE", "LTR", "unique/low-repeat", "DNA", "Simple_repeat"]
               if (rw.repeat_class == c).sum() >= 30]
        violin_box(axes[2], [rw[rw.repeat_class == c].bz2_bpb.values for c in rcl],
                   rcl, ["#c6c1e0"] * len(rcl))
        axes[2].axhline(by_class.loc["alpha_hor", "bz2_bpb"] if "alpha_hor" in by_class.index
                        else 0.73, color="#d62728", ls="--", lw=1.2,
                        label="α-satellite HOR (~0.7)")
        axes[2].set_ylabel("bzip2 bits/base"); axes[2].tick_params(axis="x", rotation=30)
        axes[2].set_title("'Non-satellite' = interspersed repeats,\nall high-complexity")
        axes[2].legend(fontsize=8)
    for k in range(3):
        axes[k].text(-0.06, 1.04, "abc"[k], transform=axes[k].transAxes,
                     fontweight="bold", fontsize=12)
    fig.suptitle("T2T-CHM13 genome-wide complexity atlas (all 24 chromosomes, 311,715 windows)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_genome_atlas.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
