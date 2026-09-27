"""Emit the deposited supplementary tables S5-S7 from the master tables.

Until now these three deposits were hand-copied from the MASTER_* files, which is
how the ape ratio column came to be rounded to one decimal in the deposit while
Table 2 of the manuscript printed it to two. This script makes the copy step
reproducible and checks, before writing anything, that each master table still
supports every count the manuscript states about it.

  results/tables/MASTER_human_all_arrays.csv -> TableS5_human_all_arrays.csv
  results/tables/MASTER_human_19chrom.csv    -> TableS6_human_per_chromosome.csv
  results/tables/MASTER_ape_47chrom.csv      -> TableS7_ape_per_chromosome.csv

The masters themselves come from build_human_arrays_all.py (S5) and
build_master_tables.py (S6, S7); run those first if the upstream analysis changed.

Every CHECK below is a claim made in the manuscript text, a table caption or a
figure caption. A failure means the deposit and the paper have diverged, so the
script writes nothing and reports which claim broke.
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")

SPEC = [
    ("MASTER_human_all_arrays.csv", "TableS5_human_all_arrays.csv",
     "complete human array classification, one row per annotated HOR array"),
    ("MASTER_human_19chrom.csv", "TableS6_human_per_chromosome.csv",
     "per-chromosome human results, one row per testable chromosome"),
    ("MASTER_ape_47chrom.csv", "TableS7_ape_per_chromosome.csv",
     "per-chromosome ape results, one row per eligible ape chromosome"),
]

OUT = []
FAILED = []


def log(m=""):
    print(m)
    OUT.append(str(m))


def check(claim, ok):
    """Record one manuscript claim and whether the master table still supports it."""
    log(f"  [{'ok ' if ok else 'FAIL'}] {claim}")
    if not ok:
        FAILED.append(claim)


def check_s5(d):
    log("Table S5 - claims made in the Supplementary Tables section:")
    check("114 annotated alpha-satellite HOR arrays", len(d) == 114)
    check("across the 19 testable chromosomes", d.chrom.nunique() == 19)
    check("81 arrays pass the >=10 kbp / >=5-window threshold",
          int(d.passes_legacy_filter.sum()) == 81)
    check("seven arrays on chr16, five below the threshold",
          (d.chrom == "chr16").sum() == 7
          and int((~d[d.chrom == "chr16"].passes_legacy_filter).sum()) == 5)
    named = ["hor_2_1", "hor_2_2", "hor_2_3", "hor_2_5", "hor_12_1",
             "hor_16_1", "hor_16_6", "hor_16_7"]
    check("contains every relict array named in the caption",
          all(d.array_name.str.startswith(n).any() for n in named))
    check("exactly one CENP-A rank-1 array per chromosome",
          (d[d.cenpa_rank == 1].groupby("chrom").size() == 1).all())


def check_s6(d, s5):
    log("Table S6 - claims made in Table 1 and its caption:")
    check("one row per testable chromosome", len(d) == 19 and d.chrom.nunique() == 19)
    check("candidate counts total 81 across the 19 chromosomes",
          int(d.n_arrays.sum()) == 81)
    check("longest-array rule correct on 19 of 19", int(d.length_hit.sum()) == 19)
    check("most-homogeneous rule correct on 17 of 19",
          int(d.homogeneity_hit.sum()) == 17)
    check("CentroSeek correct on 17 of 19", int(d.model_hit.sum()) == 17)
    check("both homogeneity misses are chr3 and chr20",
          sorted(d[~d.homogeneity_hit].chrom) == ["chr20", "chr3"])
    check("CentroSeek misses the same two chromosomes",
          sorted(d[~d.model_hit].chrom) == ["chr20", "chr3"])
    check("chr3 and chr4 carry the narrowest CENP-A margins",
          sorted(d.nsmallest(2, "cenpa_margin_over_runner_up").chrom) == ["chr3", "chr4"])
    top = s5[s5.cenpa_rank == 1].set_index("chrom").array_name
    check("CENP-A array agrees with the rank-1 array in Table S5",
          all(top[r.chrom] == r.cenpa_array for _, r in d.iterrows()))
    # Table 1's caption claims the length rule still wins over all 114 arrays,
    # not merely over the 81 that pass the size filter.
    longest = s5.loc[s5.groupby("chrom").span_bp.idxmax()].set_index("chrom").array_name
    check("length rule is 19 of 19 over all 114 annotated arrays too",
          all(longest[c] == top[c] for c in top.index))


def check_s7(d):
    log("Table S7 - claims made in Table 2, Fig. 2e and Fig. 2f:")
    check("47 eligible ape chromosomes", len(d) == 47)
    check("19 chimpanzee, 15 bonobo, 13 gorilla",
          d.species.value_counts().to_dict() == {"chimp": 19, "bonobo": 15, "gorilla": 13})
    check("15 chromosomes flagged as the published subset",
          int(d.in_published_15.sum()) == 15)
    check("longest-array rule correct on 44 of 47", int(d.length_hit.sum()) == 44)
    check("most-homogeneous rule correct on 35 of 47",
          int(d.homogeneity_hit.sum()) == 35)
    check("CentroSeek correct on 38 of 47", int(d.model_hit.sum()) == 38)
    check("all three length failures are bonobo",
          set(d[~d.length_hit].species) == {"bonobo"})
    # Fig. 2f: the ratio column must carry enough precision to distinguish
    # bonobo chr14 from a tie, which is what the one-decimal deposit could not do.
    r = d.size_ratio_active_vs_rival
    check("34 of 47 active arrays exceed their largest rival by more than 20x",
          int((r > 20).sum()) == 34)
    check("exactly three chromosomes at or below parity", int((r <= 1).sum()) == 3)
    check("the sub-parity chromosomes are the length-rule failures",
          set(d[r <= 1].chrom) == set(d[~d.length_hit].chrom))
    check("bonobo chr14 resolves as 0.98, not a 1.0 tie",
          abs(float(r[d.chrom.str.startswith("chr14_mat")].iloc[0]) - 0.9835) < 5e-4)
    check("CentroSeek recovers every sub-parity chromosome",
          bool(d[r <= 1].model_hit.all()))


def main():
    tables = {}
    for src, _, _ in SPEC:
        path = os.path.join(TAB, src)
        if not os.path.exists(path):
            sys.exit(f"missing master table {src}; run build_master_tables.py first")
        tables[src] = pd.read_csv(path)

    s5 = tables["MASTER_human_all_arrays.csv"]
    check_s5(s5)
    log()
    check_s6(tables["MASTER_human_19chrom.csv"], s5)
    log()
    check_s7(tables["MASTER_ape_47chrom.csv"])
    log()

    if FAILED:
        log(f"{len(FAILED)} claim(s) no longer hold; nothing written:")
        for c in FAILED:
            log(f"  - {c}")
        sys.exit(1)

    for src, dst, what in SPEC:
        tables[src].to_csv(os.path.join(TAB, dst), index=False)
        log(f"wrote {dst}  ({len(tables[src])} rows) - {what}")

    with open(os.path.join(ROOT, "results", "supplementary_tables_log.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
