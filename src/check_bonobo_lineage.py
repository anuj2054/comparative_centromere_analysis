"""Is the longest-array rule's failure specific to bonobo, or chance?

The manuscript states that all three failures of the longest-array rule fall in bonobo
and calls that a lineage-specific departure. That is an observation, not a test, and a
reviewer is entitled to ask whether three failures landing in one of three species is
anything more than where the dice fell.

Two tests here, on the primary haplotypes and then on both.

The both-haplotype version is the stronger evidence and is not currently used anywhere.
CenSat annotates each ape genome's maternal/paternal or hap1/hap2 assemblies separately,
so every eligible chromosome is scored twice from independently assembled sequence. If
bonobo's small active arrays were an assembly artefact they should not survive that.

Input:  results/tables/ape_censat_census.csv (eligibility and longest_is_active per
        chromosome-haplotype) and MASTER_ape_47chrom.csv (primary haplotypes)
Output: results/tables/bonobo_lineage_test.csv, results/bonobo_lineage_log.txt
"""
import os

import pandas as pd
from scipy.stats import fisher_exact

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
OUT = []


def log(m=""):
    print(m)
    OUT.append(str(m))


def test(fail_b, n_b, fail_o, n_o, label):
    odds, p = fisher_exact([[fail_b, n_b - fail_b], [fail_o, n_o - fail_o]])
    log(f"  {label:<26} bonobo {fail_b}/{n_b} failures, other apes {fail_o}/{n_o}"
        f"   Fisher exact p = {p:.2e}")
    return {"analysis": label, "bonobo_fail": fail_b, "bonobo_n": n_b,
            "other_fail": fail_o, "other_n": n_o, "fisher_p": p}


def main():
    rows = []

    a = pd.read_csv(os.path.join(TAB, "MASTER_ape_47chrom.csv"))
    b, o = a[a.species == "bonobo"], a[a.species != "bonobo"]
    log("PRIMARY HAPLOTYPES (the 47 chromosomes reported in the main text)")
    for sp, g in a.groupby("species"):
        log(f"    {sp:<9} longest-array rule {g.length_hit.sum()}/{len(g)}")
    rows.append(test(int((~b.length_hit).sum()), len(b),
                     int((~o.length_hit).sum()), len(o), "primary haplotype"))

    c = pd.read_csv(os.path.join(TAB, "ape_censat_census.csv"))
    e = c[(c.n_active > 0) & (c.n_inactive > 0)]
    b, o = e[e.species == "bonobo"], e[e.species != "bonobo"]
    log("\nBOTH HAPLOTYPES (each eligible chromosome scored twice, independently assembled)")
    for sp, g in e.groupby("species"):
        log(f"    {sp:<9} longest is active on {g.longest_is_active.sum()}/{len(g)}")
    rows.append(test(int((~b.longest_is_active).sum()), len(b),
                     int((~o.longest_is_active).sum()), len(o), "both haplotypes"))

    log("\nThe bonobo failures, by locus:")
    f = b[~b.longest_is_active].sort_values("chrom")
    for _, r in f.iterrows():
        log(f"    {r.chrom:<20} active {r.active_bp_max/1000:>8.1f} kb"
            f"   largest rival {r.inactive_bp_max/1000:>8.1f} kb")

    # a failure that recurs in both haplotypes of one chromosome is not assembly noise
    base = f.chrom.str.replace(r"_(mat|pat|hap1|hap2)_", "_", regex=True)
    both = [k for k, n in base.value_counts().items() if n > 1]
    log(f"\n  loci failing in BOTH haplotypes: {', '.join(both) if both else 'none'}")
    log("  a locus that fails in two independently assembled haplotypes is a property of")
    log("  the sequence rather than of either assembly.")

    pd.DataFrame(rows).to_csv(os.path.join(TAB, "bonobo_lineage_test.csv"), index=False)
    with open(os.path.join(ROOT, "results", "bonobo_lineage_log.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
