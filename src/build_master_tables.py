"""Build the two authoritative master tables underlying every claim in the paper.

  results/tables/MASTER_human_19chrom.csv   one row per human chromosome
  results/tables/MASTER_ape_47chrom.csv     one row per eligible ape chromosome

Both carry the per-chromosome facts, all three predictors' calls, and a difficulty
measure (the size ratio between the true active array and its largest rival), because
headline accuracies are dominated by lopsided contests and mean little without it.

Human CENP-A is CHM13-native (PRJNA559484, SRR15395851 + SRR15395853), verified against
raw BAM read density at Spearman 0.995 with identical per-chromosome calls.
Ape grading is each species' own CenSat active_hor annotation; no CENP-A exists there.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")

# the human-trained classifier, as shipped in centroseek/model.json
MEAN, SCALE = (1.939254, 10.171459), (0.161665, 0.310947)
COEF, INTER = (-0.551213, -1.405428), -0.866193


def model_p(bz2, h11):
    z = sum(((v - m) / s) * c for v, m, s, c in zip((bz2, h11), MEAN, SCALE, COEF))
    return 1.0 / (1.0 + np.exp(-(z + INTER)))


def difficulty(ratio):
    if ratio < 2:   return "hard (<2x)"
    if ratio < 5:   return "moderate (2-5x)"
    if ratio < 20:  return "easy (5-20x)"
    return "trivial (>20x)"


# ---------------------------------------------------------------- human ----
a = pd.read_csv(os.path.join(TAB, "array_level_19chrom.csv"))
a["cenpa_active"] = False
for _, g in a.groupby("chrom"):
    a.loc[g.cenpa.idxmax(), "cenpa_active"] = True
a["model_p"] = [model_p(r.bz2, r.H11) for _, r in a.iterrows()]

rows = []
for ch, g in a.groupby("chrom", sort=False):
    truth = g.loc[g.cenpa.idxmax()]
    others = g[g.index != truth.name]
    runner = others.loc[others.cenpa.idxmax()] if len(others) else None
    biggest_rival = others.span_bp.max() if len(others) else np.nan
    ratio = truth.span_bp / biggest_rival if len(others) else np.inf
    hom, ln, mo = g.loc[g.bz2.idxmin()], g.loc[g.span_bp.idxmax()], g.loc[g.model_p.idxmax()]
    rows.append(dict(
        chrom=ch, n_arrays=len(g),
        n_censat_active=int(g.censat_active.sum()),
        cenpa_array=truth["name"], cenpa_span_kb=round(truth.span_bp / 1e3, 1),
        cenpa_is_censat_active=bool(truth.censat_active),
        cenpa_margin_over_runner_up=round(truth.cenpa / runner.cenpa, 2) if runner is not None else np.nan,
        largest_rival_kb=round(biggest_rival / 1e3, 1) if len(others) else np.nan,
        size_ratio_active_vs_rival=round(ratio, 4) if np.isfinite(ratio) else np.nan,
        difficulty=difficulty(ratio),
        homogeneity_call=hom["name"], homogeneity_hit=bool(hom.cenpa_active),
        length_call=ln["name"],       length_hit=bool(ln.cenpa_active),
        model_call=mo["name"],        model_hit=bool(mo.cenpa_active),
        bz2_active=round(truth.bz2, 4), bz2_best_rival=round(others.bz2.min(), 4) if len(others) else np.nan,
    ))
H = pd.DataFrame(rows).sort_values("chrom", key=lambda s: s.str[3:].astype(int))
H.to_csv(os.path.join(TAB, "MASTER_human_19chrom.csv"), index=False)

# ------------------------------------------------------------------ ape ----
arr = pd.read_csv(os.path.join(TAB, "ape_expanded_arrays.csv"))
arr["model_p"] = [model_p(r.bz2, r.H11) for _, r in arr.iterrows()]
rows = []
for (sp, ch), g in arr.groupby(["species", "chrom"]):
    if g.censat_active.nunique() < 2:
        continue
    act, rel = g[g.censat_active], g[~g.censat_active]
    ratio = act.span_bp.max() / rel.span_bp.max()
    hom, ln, mo = g.loc[g.bz2.idxmin()], g.loc[g.span_bp.idxmax()], g.loc[g.model_p.idxmax()]
    rows.append(dict(
        species=sp, chrom=ch, in_published_15=bool(g.in_published_set.iloc[0]),
        n_arrays=len(g), n_active_hor=len(act), n_dhor=len(rel),
        active_span_kb=round(act.span_bp.max() / 1e3, 1),
        largest_dhor_kb=round(rel.span_bp.max() / 1e3, 1),
        size_ratio_active_vs_rival=round(ratio, 4),
        difficulty=difficulty(ratio),
        homogeneity_hit=bool(hom.censat_active),
        length_hit=bool(ln.censat_active),
        model_hit=bool(mo.censat_active),
        bz2_best_active=round(act.bz2.min(), 4), bz2_best_dhor=round(rel.bz2.min(), 4),
    ))
A = pd.DataFrame(rows).sort_values(["species", "chrom"])
A.to_csv(os.path.join(TAB, "MASTER_ape_47chrom.csv"), index=False)

# --------------------------------------------------------------- report ----
def block(df, tag, cols):
    print(f"\n{'='*94}\n{tag}\n{'='*94}")
    print(df[cols].to_string(index=False))

print(f"human rows {len(H)}   ape rows {len(A)}")
for name, df in [("HUMAN", H), ("APE", A)]:
    print(f"\n--- {name}: accuracy by difficulty (size ratio active vs largest rival) ---")
    order = ["hard (<2x)", "moderate (2-5x)", "easy (5-20x)", "trivial (>20x)"]
    for d in order:
        s = df[df.difficulty == d]
        if not len(s):
            continue
        print(f"  {d:17s} n={len(s):3d}   homogeneity {s.homogeneity_hit.sum():3d}/{len(s):<3d}"
              f"   length {s.length_hit.sum():3d}/{len(s):<3d}   model {s.model_hit.sum():3d}/{len(s):<3d}")
    print(f"  {'ALL':17s} n={len(df):3d}   homogeneity {df.homogeneity_hit.sum():3d}/{len(df):<3d}"
          f"   length {df.length_hit.sum():3d}/{len(df):<3d}   model {df.model_hit.sum():3d}/{len(df):<3d}")
    print(f"  median size ratio: {df.size_ratio_active_vs_rival.median():.1f}x")
