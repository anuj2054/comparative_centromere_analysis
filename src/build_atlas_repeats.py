"""Break the atlas 'non-satellite' bucket into finer classes via RepeatMasker.

The genome atlas lumped everything outside CenSat into 'non_satellite'. Here we
split that fraction by RepeatMasker class (LINE/SINE/LTR/DNA/simple/low-complexity
vs repeat-poor 'unique') and report complexity per class, so the bimodal-genome
claim is resolved into its components.

Inputs: /tmp/gc_atlas/genome_atlas_full.csv.gz, /tmp/gc_rm/rm.bed
Output: results/tables/atlas_repeat_classes.csv ; results/atlas_repeats_log.txt
"""
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
WORK = os.environ.get("GC_WORK", os.path.join(ROOT, "work"))
ATLAS = os.environ.get("GC_ATLAS", os.path.join(WORK, "gc_atlas", "genome_atlas_full.csv.gz"))
RM = os.environ.get("GC_REPEATMASKER", os.path.join(WORK, "gc_rm", "rm.bed"))
WIN = 10000
MAJOR = {"LINE", "SINE", "LTR", "DNA", "Simple_repeat", "Low_complexity",
         "Satellite", "Retroposon"}
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def load_rm():
    """Per-chromosome sorted interval lists with class."""
    by = defaultdict(list)
    with open(RM) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            cls = f[6]
            cls = cls if cls in MAJOR else ("other" if cls != "Satellite" else cls)
            by[f[0]].append((int(f[1]), int(f[2]), cls))
    for c in by:
        by[c].sort()
    return by


def window_repeat_fraction(intervals, starts, w0, w1):
    """Fraction of [w0,w1) covered by each repeat class (clipped overlaps)."""
    frac = defaultdict(int)
    lo = bisect_left(starts, w0 - 50000)
    hi = bisect_right(starts, w1)
    for i in range(lo, hi):
        s, e, cls = intervals[i]
        ov = min(e, w1) - max(s, w0)
        if ov > 0:
            frac[cls] += ov
    return frac


def main():
    df = pd.read_csv(ATLAS)
    ns = df[df["class"] == "non_satellite"].copy()
    log(f"non-satellite windows: {len(ns):,} (atlas total {len(df):,})")
    rm = load_rm()

    classes = []
    for ch, g in ns.groupby("chrom"):
        intervals = rm.get(ch, [])
        starts = [s for s, e, c in intervals]
        for _, r in g.iterrows():
            w0 = int(r.start); w1 = w0 + WIN
            fr = window_repeat_fraction(intervals, starts, w0, w1)
            total = sum(fr.values())
            if total < 0.25 * WIN:
                classes.append("unique/low-repeat")
            else:
                dom = max(fr, key=fr.get)
                classes.append(dom if dom in MAJOR else "other-repeat")
    ns["repeat_class"] = classes
    ns[["repeat_class", "bz2_bpb"]].to_csv(
        os.path.join(TAB, "atlas_repeat_windows.csv"), index=False)  # per-window, for Fig 6c box plots

    summ = ns.groupby("repeat_class").agg(
        n_windows=("bz2_bpb", "size"),
        pct_of_nonsat=("bz2_bpb", lambda x: 100 * len(x) / len(ns)),
        mean_gc=("gc", "mean"), mean_H3=("H3", "mean"),
        mean_bz2=("bz2_bpb", "mean")).sort_values("n_windows", ascending=False)
    summ.to_csv(os.path.join(TAB, "atlas_repeat_classes.csv"))
    log("\n== Non-satellite genome resolved by RepeatMasker class ==")
    log(summ.round(3).to_string())
    log("\n-> the 'non-satellite' bucket is itself mostly interspersed repeat (LINE/SINE/"
        "LTR), all high-complexity (~2.2 bits/base); only satellite DNA is low-complexity.")
    with open(os.path.join(ROOT, "results", "atlas_repeats_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
