"""Generalization test: run the complexity pipeline on centromere-spanning
regions of 5 additional chromosomes (the other acrocentrics + chrX), to test
whether the chr21 findings hold genome-wide.

Centromere centers were located from the CenSat v2.1 active-HOR spans. We fetch a
4 Mbp region around each center from NCBI (real CHM13 sequence), compute the full
metric set per 2 kbp window, then attach real Fiber-seq accessibility and CenSat
labels. Evo 2 is NOT included here (GPU-gated); this tests the CPU metrics.

Output: results/tables/multichrom_windows.csv  (pooled, with 'chrom' column)
"""
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd
import pyBigWig

sys.path.insert(0, os.path.dirname(__file__))
from lib_windows import WindowConfig, window_metrics  # noqa: E402
from fetch_tracks import collapse_censat  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BASE = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/"

# chrom -> (RefSeq acc, UCSC name, HOR center). 4 Mbp window around each center.
# CHM13 RefSeq accessions: chrN = NC_0609{24+N}.1; chrX is the 23rd record
# (NC_060947.1). NC_060948.1 is chrY (HG002/NA24385), a different individual.
CENTROMERES = {
    "chr13": ("NC_060937.1", "chr13", 14_332_063),
    "chr14": ("NC_060938.1", "chr14", 10_077_085),
    "chr15": ("NC_060939.1", "chr15", 16_555_223),
    "chr22": ("NC_060946.1", "chr22", 12_100_152),
    "chrX":  ("NC_060947.1", "chrX",  59_373_479),
}
HALF = 2_000_000


def fetch_region(acc, start, stop, chunk=500_000):
    pieces = []
    s = start
    while s <= stop:
        e = min(s + chunk - 1, stop)
        url = EFETCH + "?" + urllib.parse.urlencode(dict(
            db="nuccore", id=acc, rettype="fasta", retmode="text",
            seq_start=s, seq_stop=e))
        req = urllib.request.Request(url, headers={"User-Agent": "GenomeComplexity/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            txt = r.read().decode()
        pieces.append("".join(l.strip() for l in txt.splitlines()
                              if not l.startswith(">")))
        time.sleep(0.4)
        s = e + 1
    return "".join(pieces)


def main():
    cfg = WindowConfig(window=2000, step=4000, k_high=11, r_n_mc=40)
    bw = pyBigWig.open(BASE + "all.percent.accessible.bw")
    bb = pyBigWig.open(BASE + "censat_v2.1.bb")

    frames = []
    for ch, (acc, ucsc, center) in CENTROMERES.items():
        start = max(1, center - HALF)
        stop = center + HALF
        print(f"[{ch}] fetching {acc}:{start:,}-{stop:,} ...", flush=True)
        seq = fetch_region(acc, start, stop)
        print(f"[{ch}] {len(seq):,} bp; computing window metrics ...", flush=True)
        t = time.time()
        df = window_metrics(seq, cfg, start_offset=start)
        df.insert(0, "chrom", ch)
        # attach functional signal
        clen = bb.chroms()[ucsc]
        acc_vals, cls = [], []
        for _, r in df.iterrows():
            s0 = max(0, int(r["start"]) - 1)
            e0 = min(clen, int(r["end"]) - 1)
            v = bw.stats(ucsc, s0, e0, type="mean")[0]
            acc_vals.append(float(v) if v is not None else float("nan"))
            mid = (s0 + e0) // 2
            ents = bb.entries(ucsc, mid, mid + 1)
            name = ents[0][2].split("\t")[0] if ents else "none"
            cls.append(collapse_censat(name))
        df["accessibility"] = acc_vals
        df["censat_class"] = cls
        frames.append(df)
        print(f"[{ch}] {len(df)} windows in {time.time()-t:.0f}s "
              f"(censat: {pd.Series(cls).value_counts().to_dict()})", flush=True)

    bw.close(); bb.close()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(TAB, "multichrom_windows.csv"), index=False)
    print(f"\nwrote multichrom_windows.csv: {len(out)} windows across "
          f"{out.chrom.nunique()} chromosomes")


if __name__ == "__main__":
    main()
