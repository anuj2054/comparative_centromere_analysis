"""Attach REAL functional/annotation signal to the chr21 map windows.

Sources (T2T-CHM13 v2.0, public AWS open-data bucket human-pangenomics):
  * Fiber-seq chromatin accessibility  : browser/bbi/all.percent.accessible.bw
  * CenSat v2.1 satellite annotation   : browser/bbi/censat_v2.1.bb

For each window in results/tables/map_windows_evo2.csv (which already carries H, R,
compression, and real Evo 2 surprise), we add:
  * accessibility  : mean Fiber-seq percent-accessible over the window
  * censat_raw     : CenSat feature name covering the window midpoint
  * censat_class   : collapsed family (active_hor / other_alpha / hsat / bsat /
                     other_sat / non_satellite)

Coordinates: map_windows coords are 1-based starts on NC_060945.1 == CHM13 chr21
== UCSC 'chr21'; bigWig/bigBed are 0-based half-open, so we query [start-1, end-1].
"""
import os
import re

import pandas as pd
import pyBigWig

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
BASE = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/browser/bbi/"
ACC = BASE + "all.percent.accessible.bw"
CEN = BASE + "censat_v2.1.bb"
CHROM = "chr21"


def collapse_censat(name: str) -> str:
    """Map a CenSat v2.1 feature name to a biologically meaningful class,
    based on the family prefix used in the annotation."""
    n = name.lower()
    if n.startswith("ct"):
        return "non_satellite"          # inter-satellite transition / unique-ish
    if n.startswith("hor"):
        return "alpha_hor"              # alpha-satellite higher-order repeat (centromere core)
    if n.startswith("mon"):
        return "alpha_mono"             # monomeric alpha-satellite (pericentromeric)
    if n.startswith("hsat"):
        return "hsat"                   # human satellite (HSat1/2/3)
    if n.startswith("gsat"):
        return "gsat"
    if n.startswith("bsat"):
        return "bsat"
    if "sst1" in n or n.startswith("censat"):
        return "other_sat"
    return "non_satellite"


def main():
    df = pd.read_csv(os.path.join(TAB, "map_windows_evo2.csv"))
    bw = pyBigWig.open(ACC)
    bb = pyBigWig.open(CEN)
    clen = bb.chroms()[CHROM]

    acc, raw, cls = [], [], []
    for _, r in df.iterrows():
        s0 = max(0, int(r["start"]) - 1)
        e0 = min(clen, int(r["end"]) - 1)
        # accessibility mean (None -> NaN)
        v = bw.stats(CHROM, s0, e0, type="mean")[0]
        acc.append(float(v) if v is not None else float("nan"))
        # censat feature at window midpoint
        mid = (s0 + e0) // 2
        ents = bb.entries(CHROM, mid, mid + 1)
        if ents:
            name = ents[0][2].split("\t")[0]
        else:
            name = "none"
        raw.append(name)
        cls.append(collapse_censat(name))
    bw.close(); bb.close()

    df["accessibility"] = acc
    df["censat_raw"] = raw
    df["censat_class"] = cls
    out = os.path.join(TAB, "map_windows_functional.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} windows")
    print("\naccessibility: min %.2f mean %.2f max %.2f (NaN: %d)" % (
        df.accessibility.min(), df.accessibility.mean(), df.accessibility.max(),
        df.accessibility.isna().sum()))
    print("\ncensat_class counts:")
    print(df.censat_class.value_counts().to_string())
    print("\nraw censat families (top):")
    fams = pd.Series([re.sub(r"_\d+.*", "", x) for x in df.censat_raw]).value_counts()
    print(fams.head(12).to_string())


if __name__ == "__main__":
    main()
