"""The cross-species test on every eligible ape chromosome.

The published result used 15 hardcoded chromosomes. The CenSat census
(build_ape_censat_census.py) shows 47 chromosomes on the primary haplotypes carry both
active_hor and dhor, so only 32% of the available evidence was used. This script
scores all 47 and reports per chromosome, which is what Reviewer 1 (comment 5) and
Reviewer 2 (comment 7) ask for.

Method note: windows are tiled from each array's own start rather than from a
region origin spanning the whole centromere. Array-mean metrics are unaffected;
this only avoids fetching the megabases of non-HOR sequence between distant arrays.
Arrays are kept if they yield >= MIN_WIN windows. Because Reviewer 1's bonobo point
concerns very small active arrays, results are reported both with and without that
filter.

Output: results/tables/ape_expanded_arrays.csv
        results/tables/ape_expanded_chrom.csv
        results/ape_expanded_log.txt
"""
import os
import sys

import numpy as np
import pandas as pd
import pyBigWig
import pysam
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean, gc_content  # noqa: E402
from lib_entropy_rank import kmer_entropy  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
GA = "https://genomeark.s3.amazonaws.com/species/"
SPECIES = {
    "chimp": dict(
        censat=GA + "Pan_troglodytes/mPanTro3/assembly_curated/repeats/mPanTro3_v2.0_CenSat_v1.2.bb",
        fasta=GA + "Pan_troglodytes/mPanTro3/assembly_curated/intermediates/mPanTro3.hap1.cur.20231031.fasta.gz",
        hap="hap1"),
    "gorilla": dict(
        censat=GA + "Gorilla_gorilla/mGorGor1/assembly_curated/repeats/mGorGor1_v2.0_CenSat_v1.2.bb",
        fasta=GA + "Gorilla_gorilla/mGorGor1/assembly_curated/intermediates/mGorGor1.mat.cur.20231031.fasta.gz",
        hap="mat"),
    "bonobo": dict(
        censat=GA + "Pan_paniscus/mPanPan1/assembly_curated/repeats/mPanPan1_v2.0_CenSat_v1.2.bb",
        fasta=GA + "Pan_paniscus/mPanPan1/assembly_curated/intermediates/mPanPan1.mat.cur.20231031.fasta.gz",
        hap="mat"),
}
WINDOW, STEP, KHIGH, MIN_WIN = 2000, 4000, 11, 5
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def arrays_on(bb, ch):
    out = []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0].lower()
        if n.startswith("active_hor"):
            out.append({"start": s, "end": e, "active": True})
        elif n.startswith("dhor"):
            out.append({"start": s, "end": e, "active": False})
    return sorted(out, key=lambda a: a["start"])


def score_array(fa, ch, a):
    seq = fa.fetch(ch, a["start"], a["end"])
    bz, hh, gc = [], [], []
    for w0 in range(0, len(seq) - WINDOW + 1, STEP):
        c = clean(seq[w0:w0 + WINDOW])
        if len(c) < 0.5 * WINDOW:
            continue
        bz.append(bz2_bits_per_base(c)); hh.append(kmer_entropy(c, KHIGH)); gc.append(gc_content(c))
    if len(bz) < 1:
        return None
    return dict(n_win=len(bz), bz2=float(np.mean(bz)), H11=float(np.mean(hh)),
                gc=float(np.mean(gc)))


def main():
    census = pd.read_csv(os.path.join(TAB, "ape_censat_census.csv"))
    census["hap"] = census.chrom.str.extract(r"_(hap1|hap2|mat|pat)_")[0]

    rows = []
    for sp, cfg in SPECIES.items():
        elig = census[(census.species == sp) & (census.hap == cfg["hap"])]
        log(f"\n== {sp}: {len(elig)} eligible chromosomes on {cfg['hap']} ==")
        bb = pyBigWig.open(cfg["censat"]); fa = pysam.FastaFile(cfg["fasta"])
        for _, r in elig.iterrows():
            for ai, a in enumerate(arrays_on(bb, r.chrom)):
                m = score_array(fa, r.chrom, a)
                if m is None:
                    continue
                rows.append(dict(species=sp, chrom=r.chrom, array=ai,
                                 start=a["start"], end=a["end"],
                                 span_bp=a["end"] - a["start"],
                                 censat_active=a["active"],
                                 in_published_set=bool(r.in_published_set), **m))
            log(f"  {r.chrom:24s} arrays scored")
        bb.close(); fa.close()
    arr = pd.DataFrame(rows)
    arr.to_csv(os.path.join(TAB, "ape_expanded_arrays.csv"), index=False)
    log(f"\narrays scored: {len(arr)} across {arr.chrom.nunique()} chromosomes")

    # human-trained zero-shot classifier (trained on the corrected 19-chrom set)
    hum = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
    sc = StandardScaler().fit(hum[["bz2", "H11"]].values)
    clf = LogisticRegression(max_iter=3000).fit(
        sc.transform(hum[["bz2", "H11"]].values), hum.censat_active.astype(int))

    out = []
    for filt, tag in [(1, "all arrays"), (MIN_WIN, f">= {MIN_WIN} windows")]:
        d = arr[arr.n_win >= filt]
        for (sp, ch), g in d.groupby(["species", "chrom"]):
            if g.censat_active.nunique() < 2:
                continue
            p = clf.predict_proba(sc.transform(g[["bz2", "H11"]].values))[:, 1]
            out.append(dict(
                filter=tag, species=sp, chrom=ch,
                in_published_set=bool(g.in_published_set.iloc[0]),
                n_arrays=len(g),
                homogeneity_hit=bool(g.loc[g.bz2.idxmin(), "censat_active"]),
                length_hit=bool(g.loc[g.span_bp.idxmax(), "censat_active"]),
                lr_hit=bool(g.iloc[int(np.argmax(p))].censat_active),
                active_span_kb=g[g.censat_active].span_bp.max() / 1e3,
                relict_span_kb=g[~g.censat_active].span_bp.max() / 1e3))
    ch_df = pd.DataFrame(out)
    ch_df.to_csv(os.path.join(TAB, "ape_expanded_chrom.csv"), index=False)

    for tag, d in ch_df.groupby("filter"):
        log("\n" + "=" * 76)
        log(f"Zero-shot accuracy vs each species' own CenSat active_hor  [{tag}]")
        log("=" * 76)
        log(f"{'species':9s} {'set':14s} {'n':>3s}  {'homogeneity':>12s} {'length':>9s} {'human LR':>9s}")
        for sp in ["chimp", "gorilla", "bonobo"]:
            for lab, sub in [("published 15", d[(d.species == sp) & d.in_published_set]),
                             ("all eligible", d[d.species == sp])]:
                if not len(sub):
                    continue
                log(f"{sp:9s} {lab:14s} {len(sub):3d}  "
                    f"{sub.homogeneity_hit.sum():5d}/{len(sub):<6d} "
                    f"{sub.length_hit.sum():4d}/{len(sub):<4d} "
                    f"{sub.lr_hit.sum():4d}/{len(sub):<4d}")
        for lab, sub in [("published 15", d[d.in_published_set]), ("all eligible", d)]:
            log(f"{'TOTAL':9s} {lab:14s} {len(sub):3d}  "
                f"{sub.homogeneity_hit.sum():5d}/{len(sub):<6d} "
                f"{sub.length_hit.sum():4d}/{len(sub):<4d} "
                f"{sub.lr_hit.sum():4d}/{len(sub):<4d}")

    d = ch_df[ch_df["filter"] == "all arrays"]
    miss = d[~d.homogeneity_hit]
    log(f"\nChromosomes where the most-homogeneous rule FAILS: {len(miss)}/{len(d)}")
    for _, r in miss.iterrows():
        log(f"  {r.species:8s} {r.chrom:24s} {r.n_arrays} arrays  "
            f"active {r.active_span_kb:7.0f} kb / relict {r.relict_span_kb:7.0f} kb"
            f"{'  (in published set)' if r.in_published_set else ''}")

    with open(os.path.join(ROOT, "results", "ape_expanded_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
