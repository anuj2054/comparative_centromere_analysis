"""Rebuild the per-array table with NO size filter.

`src/build_human_arrays.py` keeps only HOR arrays with span >= 10 kb AND >= 5
windows (MIN_ARRAY_BP / MIN_WIN). That filter is undocumented in the manuscript
and silently removes small relict arrays -- including ones Reviewer 1 names on
chr2, chr12 and chr16. This script emits EVERY HOR array on the 19 testable
chromosomes, flagging (not removing) the ones the legacy filter would have cut.

Inputs are all local / directly downloaded; nothing is fetched from NCBI.
Output: results/tables/MASTER_human_all_arrays.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import pyBigWig
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_metrics import bz2_bits_per_base, clean, gc_content  # noqa: E402
from lib_entropy_rank import kmer_entropy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TAB = os.path.join(ROOT, "results", "tables")
BED = os.path.join(ROOT, "work", "censat", "chm13v2.0_censat_v2.1.bed")
FASTA = os.path.join(ROOT, "work", "cenpa", "chm13v2.0.fa")
CENPA_BW = os.path.join(ROOT, "work", "cenpa",
                        "chm13v2.0.chm13_CA_cutnrun_losalt_trimmed_q20.F3852.bw")
IGG_BW = os.path.join(ROOT, "work", "cenpa",
                      "chm13v2.0.chm13_IgG_cutnrun_losalt_q20.F3852.bw")
OUT = os.path.join(TAB, "MASTER_human_all_arrays.csv")

WINDOW, STEP, KHIGH = 2000, 4000, 11
MIN_ARRAY_BP, MIN_WIN = 10_000, 5          # the legacy filter, now only a FLAG
MIN_SHORT_BP = 500                         # below this we emit NaN metrics

TESTABLE = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr7", "chr10", "chr11",
            "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
            "chr19", "chr20", "chr21", "chr22"]


def active_label(name):
    """True iff the FIRST (dominant) HOR family token carries the live 'L'.

    hor_16_4(S1C16H1L)                    -> first token S1C16H1L  -> active
    hor_2_3(S2C2H2-B,S2C2H1L,S2C18H1L,..) -> first token S2C2H2-B  -> inactive
    Searching the whole string (the original bug) calls hor_2_3 active.
    """
    m = re.search(r"\((.*)\)\s*$", name)
    first = m.group(1).split(",")[0] if m else name
    return bool(re.search(r"H\d+L\b|H\d+L$", first))


def hor_arrays():
    rows = []
    with open(BED) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            ch, s, e, name = f[0], int(f[1]), int(f[2]), f[3]
            if ch in TESTABLE and name.lower().startswith("hor"):
                rows.append({"chrom": ch, "array_name": name, "start": s, "end": e})
    rows.sort(key=lambda r: (TESTABLE.index(r["chrom"]), r["start"]))
    return rows


def window_metrics(seq):
    """Mean bz2 / H11 / gc over 2 kb windows tiled at 4 kb step.

    Mirrors evaluate_predictor_transfer / build_human_arrays geometry: a window is used only
    if at least half of it is ACGT. Arrays shorter than one full window fall
    back to whole-array metrics (n_windows = 0) so the row still exists.
    """
    bz2s, h11s, gcs, n = [], [], [], 0
    for w0 in range(0, len(seq) - WINDOW + 1, STEP):
        c = clean(seq[w0:w0 + WINDOW])
        if len(c) < 0.5 * WINDOW:
            continue
        bz2s.append(bz2_bits_per_base(c))
        h11s.append(kmer_entropy(c, KHIGH))
        gcs.append(gc_content(c))
        n += 1
    if n:
        return float(np.mean(bz2s)), float(np.mean(h11s)), float(np.mean(gcs)), n
    c = clean(seq)
    if len(c) >= MIN_SHORT_BP:
        return (bz2_bits_per_base(c), kmer_entropy(c, KHIGH), gc_content(c), 0)
    return (np.nan, np.nan, np.nan, 0)


def bw_mean(bw, ch, s, e):
    try:
        v = bw.stats(ch, s, e, type="mean")[0]
    except Exception:
        v = None
    return float(v) if v is not None else np.nan


def main():
    arrs = hor_arrays()
    fa = pysam.FastaFile(FASTA)
    cp = pyBigWig.open(CENPA_BW)
    ig = pyBigWig.open(IGG_BW)

    rows = []
    for i, a in enumerate(arrs, 1):
        seq = fa.fetch(a["chrom"], a["start"], a["end"])
        bz2, h11, gc, nwin = window_metrics(seq)
        cpa = bw_mean(cp, a["chrom"], a["start"], a["end"])
        igg = bw_mean(ig, a["chrom"], a["start"], a["end"])
        span = a["end"] - a["start"]
        rows.append({
            "chrom": a["chrom"], "array_name": a["array_name"],
            "start": a["start"], "end": a["end"], "span_bp": span,
            "censat_active": active_label(a["array_name"]),
            "n_windows_2kb": nwin, "bz2": bz2, "H11": h11, "gc": gc,
            "cenpa_mean": cpa, "igg_mean": igg,
            "cenpa_over_igg": cpa / igg if (igg and igg == igg and igg != 0) else np.nan,
            "passes_legacy_filter": bool(span >= MIN_ARRAY_BP and nwin >= MIN_WIN),
        })
        print(f"[{i}/{len(arrs)}] {a['chrom']} {a['array_name']} "
              f"{span/1e3:.1f} kb nwin={nwin}", flush=True)

    fa.close(); cp.close(); ig.close()
    df = pd.DataFrame(rows)

    df["cenpa_rank"] = df.groupby("chrom")["cenpa_mean"].rank(
        ascending=False, method="min").astype("Int64")
    df["length_rank"] = df.groupby("chrom")["span_bp"].rank(
        ascending=False, method="min").astype("Int64")
    df["homogeneity_rank"] = df.groupby("chrom")["bz2"].rank(
        ascending=True, method="min").astype("Int64")

    df = df[["chrom", "array_name", "start", "end", "span_bp", "censat_active",
             "n_windows_2kb", "bz2", "H11", "gc", "cenpa_mean", "igg_mean",
             "cenpa_over_igg", "cenpa_rank", "length_rank", "homogeneity_rank",
             "passes_legacy_filter"]]
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} rows, {df.chrom.nunique()} chromosomes, "
          f"{int(df.passes_legacy_filter.sum())} pass the legacy filter")


if __name__ == "__main__":
    main()
