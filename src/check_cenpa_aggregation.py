"""How much does the functional label depend on how CENP-A is aggregated?

Four of the five human failures pair a multi-megabase CenSat-active array against a
tiny relict that CENP-A ranks higher, three of them by a margin under 1.35x. The
label uses the MEAN CENP-A signal over the array, which is not comparable between a
21 kb array and a 4.5 Mb one. This script recomputes every downstream number under
alternative aggregations.

  mean          : current definition (bigWig mean over the array)
  total         : mean x span, i.e. integrated CENP-A signal
  mean, >=100kb : mean, but only arrays >= 100 kb may be called functional
"""
import os
import numpy as np
import pandas as pd

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
a = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
a["total"] = a.cenpa * a.span_bp

DEFS = {
    "mean (current)":       lambda g: g.cenpa.idxmax(),
    "total signal":         lambda g: g.total.idxmax(),
    "mean, arrays >=100kb": lambda g: (g[g.span_bp >= 100_000].cenpa.idxmax()
                                       if (g.span_bp >= 100_000).any() else g.cenpa.idxmax()),
}
print(f"{'definition':22s} {'CenSat agree':>13s} {'homogeneity':>12s} {'length':>8s}")
for name, pick in DEFS.items():
    agree = hom = ln = n = 0
    for ch, g in a.groupby("chrom"):
        idx = pick(g)
        agree += bool(a.loc[idx, "censat_active"])
        hom += (g.bz2.idxmin() == idx)
        ln += (g.span_bp.idxmax() == idx)
        n += 1
    print(f"{name:22s} {agree:6d}/{n:<6d} {hom:6d}/{n:<5d} {ln:4d}/{n:<4d}")

print("\nchromosomes whose functional call changes between 'mean' and 'total':")
for ch, g in a.groupby("chrom"):
    m, t = g.cenpa.idxmax(), g.total.idxmax()
    if m != t:
        print(f"  {ch:7s} mean -> {a.loc[m,'name'][:34]:34s} ({a.loc[m,'span_bp']/1e3:7.0f} kb)")
        print(f"  {'':7s} total-> {a.loc[t,'name'][:34]:34s} ({a.loc[t,'span_bp']/1e3:7.0f} kb)"
              f"  {'CenSat-active' if a.loc[t,'censat_active'] else ''}")
