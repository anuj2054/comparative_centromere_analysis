"""The dinucleotide-shuffle control and the scale sweep, both of which need sequence.

(A) SHUFFLE CONTROL ON THE ACTIVE/INACTIVE CONTRAST.
    The manuscript applies dinucleotide-preserving shuffles only to the Entropy-Rank
    Ratio null and to the class-level satellite contrast, never to the active-vs-
    inactive comparison that carries the paper. If the contrast is higher-order
    repeat structure rather than composition, active and inactive windows must
    collapse to the same floor once shuffled.

(B) WINDOW SCALE, READ CORRECTLY.
    Section 3.4 reads "AUROC rising from 0.82 to 0.93 to 0.97 to 1.00 between 1 and
    50 kb" as evidence that 2 kb is conservative. A monotone rise with window size is
    instead what a redundancy-driven statistic does: a larger window admits more
    copies of the repeat unit. This recomputes the sweep on the corrected 19-chromosome
    set and normalises window size to each array's dominant self-similarity period.

Sequence is cached under the scratchpad so re-runs are cheap.

Output: results/tables/step6_shuffle_control.csv
        results/tables/step6_scale_sweep.csv
        results/shuffle_control_log.txt
"""
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pyBigWig
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean, dinucleotide_shuffle  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
WORK = os.environ.get("GC_WORK", os.path.join(ROOT, "work"))
CACHE = os.environ.get("SEQ_CACHE", os.path.join(WORK, "seqcache"))
os.makedirs(CACHE, exist_ok=True)
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CEN = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/censat_v2.1.bb"
CHROMS = [f"chr{n}" for n in range(1, 23)] + ["chrX", "chrY"]
ACC = {c: f"NC_0609{25 + i}.1" for i, c in enumerate(CHROMS)}
SCALES = [1000, 2000, 5000, 10000, 20000, 50000]
RNG = np.random.default_rng(20260818)
LOG = []


def log(m=""):
    print(m, flush=True); LOG.append(str(m))


def active_label(name):
    m = re.search(r"\((.*)\)\s*$", name)
    first = m.group(1).split(",")[0] if m else name
    return bool(re.search(r"H\d+L\b|H\d+L$", first))


def fetch(acc, lo, hi, chunk=500_000):
    key = os.path.join(CACHE, f"{acc}_{lo}_{hi}.txt")
    if os.path.exists(key):
        return open(key).read()
    out, s = [], lo
    while s <= hi:
        e = min(s + chunk - 1, hi)
        url = EFETCH + "?" + urllib.parse.urlencode(dict(
            db="nuccore", id=acc, rettype="fasta", retmode="text",
            seq_start=s, seq_stop=e))
        req = urllib.request.Request(url, headers={"User-Agent": "GC/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            txt = r.read().decode()
        out.append("".join(l.strip() for l in txt.splitlines() if not l.startswith(">")))
        time.sleep(0.34)
        s = e + 1
    seq = "".join(out)
    open(key, "w").write(seq)
    return seq


def dominant_period(seq, lo=80, hi=3000, samp=30000):
    """Peak of the one-hot autocorrelation -- the array's repeat unit length."""
    s = clean(seq[:samp])
    if len(s) < 2 * hi:
        return np.nan
    x = np.frombuffer(s.encode(), dtype=np.uint8).astype(float)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    seg = ac[lo:hi]
    return int(lo + np.argmax(seg)) if len(seg) else np.nan


def main():
    bb = pyBigWig.open(CEN)
    arr_tab = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
    testable = sorted(arr_tab.chrom.unique(), key=CHROMS.index)

    shuf_rows, scale_rows = [], []
    for ch in testable:
        spans = [(s, e, r.split("\t")[0]) for s, e, r in
                 (bb.entries(ch, 0, bb.chroms()[ch]) or [])
                 if r.split("\t")[0].lower().startswith("hor")]
        spans = [(s, e, n) for s, e, n in spans if e - s >= 10_000]
        if not spans:
            continue
        lo = max(1, min(s for s, _, _ in spans) - 1000)
        hi = max(e for _, e, _ in spans) + 1000
        seq = fetch(ACC[ch], lo, hi)
        log(f"[{ch}] {len(seq)/1e6:.2f} Mb, {len(spans)} arrays >= 10 kb")

        for s, e, name in spans:
            act = active_label(name)
            sub = seq[s - lo:e - lo]
            per = dominant_period(sub)

            # (A) shuffle control at the paper's 2 kb scale
            real, shuf = [], []
            for w0 in range(0, len(sub) - 2000 + 1, 4000):
                c = clean(sub[w0:w0 + 2000])
                if len(c) < 1000:
                    continue
                real.append(bz2_bits_per_base(c))
                shuf.append(bz2_bits_per_base(dinucleotide_shuffle(c, RNG)))
            if real:
                shuf_rows.append(dict(chrom=ch, name=name, censat_active=act,
                                      n_win=len(real), bz2_real=float(np.mean(real)),
                                      bz2_shuffled=float(np.mean(shuf))))

            # (B) scale sweep
            for W in SCALES:
                vals = []
                for w0 in range(0, len(sub) - W + 1, 2 * W):
                    c = clean(sub[w0:w0 + W])
                    if len(c) < 0.5 * W:
                        continue
                    vals.append(bz2_bits_per_base(c))
                if len(vals) >= 2:
                    scale_rows.append(dict(chrom=ch, name=name, censat_active=act,
                                           window=W, period=per,
                                           w_over_period=W / per if per and not np.isnan(per) else np.nan,
                                           n_win=len(vals), bz2=float(np.mean(vals))))
    bb.close()

    sh = pd.DataFrame(shuf_rows); sh.to_csv(os.path.join(TAB, "step6_shuffle_control.csv"), index=False)
    sc = pd.DataFrame(scale_rows); sc.to_csv(os.path.join(TAB, "step6_scale_sweep.csv"), index=False)

    log("\n" + "=" * 78)
    log("(A) Dinucleotide-shuffle control on the active/inactive contrast")
    log("=" * 78)
    for lab, g in [("active", sh[sh.censat_active]), ("inactive", sh[~sh.censat_active])]:
        log(f"  {lab:9s} n={len(g):3d} arrays   real {g.bz2_real.mean():.3f}   "
            f"shuffled {g.bz2_shuffled.mean():.3f}")
    a, i = sh[sh.censat_active], sh[~sh.censat_active]
    log(f"\n  real     active - inactive = {a.bz2_real.mean()-i.bz2_real.mean():+.3f}")
    log(f"  shuffled active - inactive = {a.bz2_shuffled.mean()-i.bz2_shuffled.mean():+.3f}")
    ya = np.r_[np.ones(len(a)), np.zeros(len(i))]
    log(f"  AUROC real     = {roc_auc_score(ya, -np.r_[a.bz2_real, i.bz2_real]):.3f}")
    log(f"  AUROC shuffled = {roc_auc_score(ya, -np.r_[a.bz2_shuffled, i.bz2_shuffled]):.3f}")

    log("\n" + "=" * 78)
    log("(B) Window-scale sweep on the corrected 19-chromosome set")
    log("=" * 78)
    log(f"  {'window':>8s} {'arrays':>7s} {'AUROC':>7s} {'act mean':>9s} {'ina mean':>9s} "
        f"{'median W/period':>16s}")
    for W, g in sc.groupby("window"):
        a, i = g[g.censat_active], g[~g.censat_active]
        if not len(a) or not len(i):
            continue
        y = np.r_[np.ones(len(a)), np.zeros(len(i))]
        au = roc_auc_score(y, -np.r_[a.bz2, i.bz2])
        log(f"  {W:8d} {len(g):7d} {au:7.3f} {a.bz2.mean():9.3f} {i.bz2.mean():9.3f} "
            f"{g.w_over_period.median():16.1f}")
    log("\n  median dominant period: active "
        f"{sc[sc.censat_active].period.median():.0f} bp, inactive "
        f"{sc[~sc.censat_active].period.median():.0f} bp")

    with open(os.path.join(ROOT, "results", "shuffle_control_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
