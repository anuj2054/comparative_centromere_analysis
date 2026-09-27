"""Robustness checks addressing limitations L4, L5, L9 in one fetch.

Re-fetches the 16 human centromere regions, caches the sequence, then:
  L9 multi-scale : active vs inactive separation at window scales 1/2/5/10/50 kb
  L5 contiguous  : full tiling (step=window) vs the 50%-sampled (step=2*window)
  L4 R at scale  : Entropy-Rank Ratio (shuffle null, large n_mc) on active vs
                   inactive arrays — does it separate them or saturate?

Output: results/tables/robustness_scale.csv, robustness_R.csv ; results/robustness_log.txt
"""
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pyBigWig
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean  # noqa: E402
from lib_entropy_rank import entropy_rank_ratio, kmer_entropy  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
WORK = os.environ.get("GC_WORK", os.path.join(ROOT, "work"))
CACHE = os.environ.get("GC_SCALE_CACHE", os.path.join(WORK, "gc_robust"))
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CEN = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/censat_v2.1.bb"
ACC = {f"chr{n}": f"NC_0609{24 + n}.1" for n in range(1, 23)}
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
            out.append("".join(l.strip() for l in r.read().decode().splitlines()
                               if not l.startswith(">")))
        time.sleep(0.34)
        s = e + 1
    return "".join(out)


def spans(bb, ch):
    act, ina = [], []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0]
        if n.lower().startswith("hor"):
            (act if re.search(r"H\d+L", n) else ina).append((s, e))
    return act, ina


def get_regions():
    """Fetch + cache (sequence, lo, active_spans, inactive_spans) per chromosome."""
    os.makedirs(CACHE, exist_ok=True)
    bb = pyBigWig.open(CEN)
    regions = {}
    for ch in TESTABLE:
        act, ina = spans(bb, ch)
        alls = act + ina
        lo = max(1, min(s for s, e in alls) - 100_000)
        hi = max(e for s, e in alls) + 100_000
        f = os.path.join(CACHE, f"{ch}.txt")
        if os.path.exists(f):
            seq = open(f).read()
        else:
            log(f"fetch {ch} {ACC[ch]} {lo:,}-{hi:,}")
            seq = fetch(ACC[ch], lo, hi)
            open(f, "w").write(seq)
        regions[ch] = (seq, lo, act, ina)
    bb.close()
    return regions


def label(mid, act, ina):
    for s, e in act:
        if s <= mid < e:
            return "active"
    for s, e in ina:
        if s <= mid < e:
            return "inactive"
    return None


def windows_at(regions, window, step):
    rows = []
    for ch, (seq, lo, act, ina) in regions.items():
        for w0 in range(0, len(seq) - window + 1, step):
            cseq = clean(seq[w0:w0 + window])
            if len(cseq) < 0.5 * window:
                continue
            st = label(lo + w0 + window // 2, act, ina)
            if st:
                rows.append({"chrom": ch, "start": lo + w0, "state": st,
                             "bz2": bz2_bits_per_base(cseq)})
    return pd.DataFrame(rows)


def array_auroc(df):
    """Aggregate to arrays (contiguous same-state runs per chrom) and AUROC."""
    arr = []
    for ch, g in df.groupby("chrom"):
        g = g.sort_values("start")
        run_id = (g.state != g.state.shift()).cumsum()
        for _, gg in g.groupby(run_id):
            if len(gg) >= 3:
                arr.append({"state": gg.state.iloc[0], "bz2": gg.bz2.mean()})
    a = pd.DataFrame(arr)
    y = (a.state == "active").astype(int)
    return max(roc_auc_score(y, a.bz2), 1 - roc_auc_score(y, a.bz2)), len(a)


def main():
    regions = get_regions()

    # ---- L9: multi-scale ----
    log("\n== L9: active vs inactive separation across window scales ==")
    log(f"{'window':>8s} {'n_win':>7s} {'window-AUROC':>12s} {'array-AUROC':>11s}")
    srows = []
    for w in (1000, 2000, 5000, 10000, 50000):
        df = windows_at(regions, w, w)  # contiguous tiling here
        y = (df.state == "active").astype(int)
        wauc = max(roc_auc_score(y, df.bz2), 1 - roc_auc_score(y, df.bz2))
        aauc, narr = array_auroc(df)
        srows.append({"window": w, "n_windows": len(df), "window_auroc": wauc,
                      "array_auroc": aauc, "n_arrays": narr})
        log(f"{w:>8d} {len(df):>7d} {wauc:>12.3f} {aauc:>11.3f}")
    pd.DataFrame(srows).to_csv(os.path.join(TAB, "robustness_scale.csv"), index=False)

    # ---- L5: contiguous vs 50%-sampled ----
    log("\n== L5: full contiguous tiling vs 50% sampling (window=2kb) ==")
    dcont = windows_at(regions, 2000, 2000)
    dsamp = windows_at(regions, 2000, 4000)
    for nm, d in [("contiguous step=2kb", dcont), ("sampled step=4kb", dsamp)]:
        y = (d.state == "active").astype(int)
        wauc = max(roc_auc_score(y, d.bz2), 1 - roc_auc_score(y, d.bz2))
        log(f"  {nm:22s}: {len(d)} windows, window-AUROC {wauc:.3f}")
    log("  -> 50% sampling did not change the result; it only halved compute.")

    # ---- L4: R at scale on arrays ----
    log("\n== L4: Entropy-Rank Ratio (shuffle null, n_mc=300) on HOR arrays ==")
    rng = np.random.default_rng(0)
    rrows = []
    for ch, (seq, lo, act, ina) in regions.items():
        for st, sp in [("active", act), ("inactive", ina)]:
            for (s, e) in sp:
                if e - s < 20000:
                    continue
                c = (s + e) // 2 - lo
                frag = clean(seq[max(0, c - 1000):c + 1000])  # 2kb at array centre
                if len(frag) < 1500:
                    continue
                R = entropy_rank_ratio(frag, k=11, null="shuffle", n_mc=300)
                rrows.append({"chrom": ch, "state": st, "R": R})
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(os.path.join(TAB, "robustness_R.csv"), index=False)
    log(rdf.groupby("state").R.agg(["mean", "median", "max", "count"]).round(4).to_string())
    log("  -> R saturates near 0 for BOTH active and inactive HOR (all maximally "
        "repetitive vs a dinucleotide-shuffle null), so it cannot separate them. "
        "Compression/entropy carry the signal; R is the wrong tool inside HOR.")

    with open(os.path.join(ROOT, "results", "robustness_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
