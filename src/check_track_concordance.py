"""Do the per-chromosome functional calls depend on the assay or the salt fraction?

Two open items close here.

R2#6 asked for stronger functional validation. The reported label comes from CENP-A
CUT&RUN, one assay with no input normalisation. build_chipseq_track.sh builds an orthogonal CENP-A ChIP-seq
track from the same cell line with its own matched input. If the ChIP-seq picks the same
array on every chromosome, the label is assay-independent.

The Methods disclosed that the label uses the low-salt fraction only and that invariance to
the high-salt or merged fractions was untested. build_highsalt_track.sh builds both. If the calls hold, that
sentence becomes a result instead of a caveat.

For each track the active array on a chromosome is the candidate with the highest mean
coverage, exactly as the reported label is defined. What matters is not whether the tracks
agree numerically but whether they select the same array.

Output: results/tables/track_concordance.csv, results/track_concordance_log.txt
"""
import os

import numpy as np
import pandas as pd
import pyBigWig

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
W = os.path.join(ROOT, "work", "cenpa")
OUT = []

TRACKS = {
    "losalt":  "chm13v2.0.chm13_CA_cutnrun_losalt_trimmed_q20.F3852.bw",   # the reported label
    "hisalt":  "chm13v2.0.chm13_CA_cutnrun_hisalt_trimmed_q20.F3852.bw",
    "merged":  "chm13v2.0.chm13_CA_cutnrun_merged_trimmed_q20.F3852.bw",
    "chipseq": "chm13v2.0.chm13_CA_chip_q20.F3852.bw",
    "input":   "chm13v2.0.chm13_CA_input_q20.F3852.bw",                    # ChIP-seq control
    "igg":     "chm13v2.0.chm13_IgG_cutnrun_losalt_q20.F3852.bw",          # CUT&RUN control
}


def log(m=""):
    print(m)
    OUT.append(str(m))


def mean_cov(bw, ch, s, e):
    try:
        v = bw.stats(ch, int(s), int(e), type="mean")[0]
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


def main():
    arr = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
    bws = {k: pyBigWig.open(os.path.join(W, f)) for k, f in TRACKS.items()}

    for k, bw in bws.items():
        arr[k] = [mean_cov(bw, r.chrom, r.start, r.end) for _, r in arr.iterrows()]
    for bw in bws.values():
        bw.close()

    rows = []
    for ch, g in arr.groupby("chrom"):
        row = {"chrom": ch, "n_arrays": len(g)}
        for k in TRACKS:
            row[k] = g.loc[g[k].idxmax(), "name"] if g[k].notna().any() else None
        rows.append(row)
    d = pd.DataFrame(rows)

    ref = "losalt"
    log(f"Per-chromosome active-array call, {len(d)} chromosomes, reference = {ref}\n")
    log(f"  {'track':<9}{'agrees with low-salt':>22}   role")
    roles = {"hisalt": "the untested salt fraction", "merged": "low+high, as the hub Methods describe",
             "chipseq": "orthogonal assay (R2#6)", "input": "ChIP-seq background, expect disagreement",
             "igg": "CUT&RUN background, expect disagreement"}
    for k in ["hisalt", "merged", "chipseq", "input", "igg"]:
        agree = (d[k] == d[ref]).sum()
        d[f"{k}_same"] = d[k] == d[ref]
        log(f"  {k:<9}{f'{agree}/{len(d)}':>22}   {roles[k]}")

    log("\nChromosomes where any CENP-A track disagrees with the reported label:")
    dis = d[~(d.hisalt_same & d.merged_same & d.chipseq_same)]
    if len(dis) == 0:
        log("  none. Every CENP-A track selects the same array on all 19 chromosomes.")
    else:
        for _, r in dis.iterrows():
            log(f"  {r.chrom}: losalt {r[ref]} | hisalt {r.hisalt} | "
                f"merged {r.merged} | chipseq {r.chipseq}")

    d.to_csv(os.path.join(TAB, "track_concordance.csv"), index=False)
    with open(os.path.join(ROOT, "results", "track_concordance_log.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
