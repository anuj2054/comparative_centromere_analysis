"""Correct the testable-chromosome set and the
active-array label rule, then rebuild the array-level table.

Two defects in the original pipeline, both raised by Reviewer 1:

R1#1 -- `evaluate_predictor_transfer.TESTABLE` is a hardcoded list of 16 chromosomes said to
be "from CenSat scan". Re-scanning CenSat v2.1 shows 19 chromosomes carry both an
active and an inactive HOR array. chr2, chr12 and chr16 were wrongly omitted. The
five genuinely active-only chromosomes are chr6, chr8, chr9, chrX and chrY, which
is exactly the set Reviewer 1 names.

R1#1b -- the active-label rule `re.search(r"H\\d+L", name)` scans the whole CenSat
feature name. Composite names list several HOR families inside the parentheses, so
a relict array whose name happens to mention a live family is mislabelled active.
Exactly one array in CHM13 is affected, and it is one Reviewer 1 names as inactive:
  hor_2_3(S2C2H2-B,S2C2H1L,S2C18H1L,S2C20H1L,S2C20H5d)
The corrected rule reads only the FIRST (dominant) family token.

Window metrics for the original 16 chromosomes are reused from
results/tables/human_all_centromeres.csv (they do not depend on the labels); only
chr2, chr12 and chr16 are fetched and scored here.

Output: results/tables/human_all_centromeres_19chrom.csv   (window level)
        results/tables/array_level_19chrom.csv             (array level)
        results/human_arrays_log.txt
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

sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean, gc_content  # noqa: E402
from lib_entropy_rank import kmer_entropy  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
B = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/"
CEN = B + "browser/bbi/censat_v2.1.bb"
# Default is the published HG002 track (a cross-individual proxy). build_cenpa_track.sh builds a
# CHM13-native track; point CENPA_BW at it to re-run everything against sample-matched
# data without editing code. The accessions are listed in the README's Data sources.
CENPA = os.environ.get(
    "CENPA_BW",
    B + "assemblies/alignments/cutnrun/chm13v2.0.hg002_CA_cutnrun_losalt_trimmed_q20_2.F3852.bw")
WINDOW, STEP, KHIGH = 2000, 4000, 11
MIN_ARRAY_BP, MIN_WIN = 10_000, 5      # unchanged from analyze_cenpa_occupancy.py

CHROMS = [f"chr{n}" for n in range(1, 23)] + ["chrX", "chrY"]
ACC = {c: f"NC_0609{25 + i}.1" for i, c in enumerate(CHROMS)}
NEW = ["chr2", "chr12", "chr16"]
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def active_label(name):
    """True if the array's DOMINANT HOR family carries the live 'L' designation.

    CenSat names look like hor_16_4(S1C16H1L) or, when several families are
    present, hor_2_3(S2C2H2-B,S2C2H1L,...). Only the first token describes the
    array itself; the rest are minor constituents. The original rule searched the
    whole string and so inherited any 'L' appearing anywhere in the list.
    """
    m = re.search(r"\((.*)\)\s*$", name)
    first = m.group(1).split(",")[0] if m else name
    return bool(re.search(r"H\d+L\b|H\d+L$", first))


def hor_arrays(bb, ch, min_bp=MIN_ARRAY_BP):
    out = []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0]
        if n.lower().startswith("hor") and (e - s) >= min_bp:
            out.append({"start": s, "end": e, "name": n,
                        "censat_active": active_label(n),
                        "censat_active_oldrule": bool(re.search(r"H\d+L", n))})
    return sorted(out, key=lambda a: a["start"])


def fetch(acc, lo, hi, chunk=500_000):
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
    return "".join(out)


def score_new_chrom(bb, ch):
    """Window metrics for a chromosome absent from the original run.

    Region and window geometry are identical to evaluate_predictor_transfer.build() so the
    new windows are commensurable with the cached ones.
    """
    arrs = hor_arrays(bb, ch, min_bp=0)          # region spans ALL HOR, as before
    lo = max(1, min(a["start"] for a in arrs) - 100_000)
    hi = max(a["end"] for a in arrs) + 100_000
    log(f"  [{ch}] {ACC[ch]} {lo:,}-{hi:,} ({(hi-lo)/1e6:.2f} Mb) -> fetching")
    seq = fetch(ACC[ch], lo, hi)
    rows = []
    for w0 in range(0, len(seq) - WINDOW + 1, STEP):
        cseq = clean(seq[w0:w0 + WINDOW])
        if len(cseq) < 0.5 * WINDOW:
            continue
        mid = lo + w0 + WINDOW // 2
        hit = [a for a in arrs if a["start"] <= mid < a["end"]]
        if not hit:
            continue
        rows.append({"chrom": ch, "start": lo + w0,
                     "state": "active" if hit[0]["censat_active"] else "inactive",
                     "gc": gc_content(cseq), "bz2_bpb": bz2_bits_per_base(cseq),
                     "H11": kmer_entropy(cseq, KHIGH)})
    log(f"  [{ch}] {len(rows)} windows in HOR arrays")
    return pd.DataFrame(rows)


def main():
    bb = pyBigWig.open(CEN)

    log("== CenSat v2.1 rescan: which chromosomes carry both array types ==")
    counts = {}
    for ch in CHROMS:
        a = hor_arrays(bb, ch, min_bp=0)
        counts[ch] = (sum(x["censat_active"] for x in a),
                      sum(not x["censat_active"] for x in a))
    testable = [c for c in CHROMS if counts[c][0] and counts[c][1]]
    log(f"  testable (both classes): {len(testable)} -> {', '.join(testable)}")
    log(f"  active-only            : {', '.join(c for c in CHROMS if not counts[c][1])}")
    flips = [(ch, a) for ch in CHROMS for a in hor_arrays(bb, ch, min_bp=0)
             if a["censat_active"] != a["censat_active_oldrule"]]
    log(f"  arrays relabelled by the corrected rule: {len(flips)}")
    for ch, a in flips:
        log(f"    {ch} {(a['end']-a['start'])/1e3:.0f} kb  "
            f"old={'active' if a['censat_active_oldrule'] else 'inactive'} -> "
            f"new={'active' if a['censat_active'] else 'inactive'}  {a['name']}")

    # ---- window table: cached 16 + newly scored 3 -------------------------
    cached = pd.read_csv(os.path.join(TAB, "human_all_centromeres.csv"))
    log(f"\n== Windows ==\n  cached: {len(cached)} across {cached.chrom.nunique()} chromosomes")
    new_path = os.path.join(TAB, "windows_chr2_12_16.csv")
    if os.path.exists(new_path):
        new = pd.read_csv(new_path)
        log(f"  new   : {len(new)} reused from {os.path.basename(new_path)}")
    else:
        new = pd.concat([score_new_chrom(bb, c) for c in NEW], ignore_index=True)
        new.to_csv(new_path, index=False)
    win = pd.concat([cached, new], ignore_index=True)
    win.to_csv(os.path.join(TAB, "human_all_centromeres_19chrom.csv"), index=False)
    log(f"  total : {len(win)} across {win.chrom.nunique()} chromosomes")

    # ---- array level ------------------------------------------------------
    cp = pyBigWig.open(CENPA)
    rows, dropped = [], []
    for ch in sorted(win.chrom.unique(), key=lambda c: CHROMS.index(c)):
        g = win[win.chrom == ch]
        for ai, a in enumerate(hor_arrays(bb, ch)):
            sub = g[(g.start + 1000 >= a["start"]) & (g.start + 1000 < a["end"])]
            if len(sub) < MIN_WIN:
                dropped.append((ch, a["name"], a["end"] - a["start"], len(sub)))
                continue
            cpa = cp.stats(ch, a["start"], a["end"], type="mean")[0]
            rows.append({"chrom": ch, "array": ai, "name": a["name"],
                         "start": a["start"], "end": a["end"],
                         "span_bp": a["end"] - a["start"], "n_win": len(sub),
                         "bz2": sub.bz2_bpb.mean(), "H11": sub.H11.mean(),
                         "gc": sub.gc.mean(), "censat_active": a["censat_active"],
                         "cenpa": cpa if cpa is not None else np.nan})
    bb.close(); cp.close()
    arr = pd.DataFrame(rows).dropna(subset=["cenpa"])
    arr.to_csv(os.path.join(TAB, "array_level_19chrom.csv"), index=False)
    log(f"\n== Arrays ==\n  kept   : {len(arr)} across {arr.chrom.nunique()} chromosomes "
        f"({arr.censat_active.sum()} CenSat-active / {(~arr.censat_active).sum()} inactive)")
    log(f"  dropped: {len(dropped)} arrays below {MIN_ARRAY_BP/1000:.0f} kb or "
        f"<{MIN_WIN} windows (this is why small p-arm relicts vanish; see R1#2)")
    for ch, n, bp, nw in dropped:
        log(f"    {ch} {bp/1e3:6.0f} kb  {nw} win  {n}")

    with open(os.path.join(ROOT, "results", "human_arrays_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
