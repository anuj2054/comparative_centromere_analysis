"""Re-derive the manuscript's numbers from the deposited tables.

    python verify/verify_claims.py

Every check below restates a number that appears in the paper and recomputes it from
results/tables/. A failure means the deposit and the manuscript have drifted apart.
This exists because that drift is not hypothetical: during revision a stale
array-level p-value from a superseded 16-chromosome analysis was carried into a
supplementary table, and a ratio column rounded to one decimal was printed to two.
Both would have been caught here.

Exit status is 0 when every claim holds and 1 otherwise, so this can gate a release.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "results", "tables")
PASS, FAIL = [], []


def T(name):
    return pd.read_csv(os.path.join(TAB, name))


def check(claim, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    (PASS if ok else FAIL).append((claim, got, want))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {claim}: {got}" + ("" if ok else f"  (expected {want})"))


def section(t):
    print(f"\n{t}")


def main():
    # ---- Table 1 and Results 3.1: the human result ----
    section("Human, 19 chromosomes (Table 1, Results 3.1)")
    s5, s6 = T("MASTER_human_all_arrays.csv"), T("MASTER_human_19chrom.csv")
    check("annotated HOR arrays", len(s5), 114)
    check("arrays passing the size threshold", int(s5.passes_legacy_filter.sum()), 81)
    check("testable chromosomes", s6.chrom.nunique(), 19)
    check("longest-array rule, top-1", int(s6.length_hit.sum()), 19)
    check("most-homogeneous rule, top-1", int(s6.homogeneity_hit.sum()), 17)
    check("CentroSeek, top-1", int(s6.model_hit.sum()), 17)
    check("CENP-A array is CenSat-active on", int(s6.cenpa_is_censat_active.sum()), 19)

    al = T("array_level_19chrom.csv").dropna(subset=["bz2"])
    act = al.groupby("chrom").cenpa.transform("max") == al.cenpa
    check("mean bzip2, CENP-A-active arrays", round(al[act].bz2.mean(), 2), 1.75, 5e-3)
    check("mean bzip2, competing arrays", round(al[~act].bz2.mean(), 2), 2.00, 5e-3)
    check("Mann-Whitney p, CENP-A label (x1e8)",
          round(mannwhitneyu(al[act].bz2, al[~act].bz2).pvalue * 1e8, 1), 1.9, 0.05)
    ca = al.censat_active.astype(bool)
    check("CenSat-labelled contrast, n active", int(ca.sum()), 27)
    check("CenSat-labelled contrast, n inactive", int((~ca).sum()), 54)
    check("Mann-Whitney p, CenSat label (x1e9)",
          round(mannwhitneyu(al[ca].bz2, al[~ca].bz2).pvalue * 1e9, 1), 9.7, 0.05)

    # ---- Fig. 1e ----
    section("Fig. 1e: length and homogeneity are coupled")
    from scipy.stats import spearmanr
    check("Spearman rho, log span vs bzip2",
          round(spearmanr(np.log10(al.span_bp), al.bz2)[0], 2), -0.51, 5e-3)
    check("smallest active array, kbp", round(al[act].span_bp.min() / 1e3), 331, 0.5)
    check("competing arrays exceeding it", int((al[~act].span_bp > al[act].span_bp.min()).sum()), 8)

    # ---- Table 2, Fig. 2e, Fig. 2f: the ape result ----
    section("Great apes, 47 chromosomes (Table 2, Fig. 2e, Fig. 2f)")
    s7 = T("MASTER_ape_47chrom.csv")
    check("eligible ape chromosomes", len(s7), 47)
    check("longest-array rule, top-1", int(s7.length_hit.sum()), 44)
    check("most-homogeneous rule, top-1", int(s7.homogeneity_hit.sum()), 35)
    check("CentroSeek, top-1", int(s7.model_hit.sum()), 38)
    for sp, n, ln in [("chimp", 19, 19), ("gorilla", 13, 13), ("bonobo", 15, 12)]:
        g = s7[s7.species == sp]
        check(f"{sp} chromosomes", len(g), n)
        check(f"{sp} longest-array top-1", int(g.length_hit.sum()), ln)
    check("length-rule failures that are bonobo",
          int((s7[~s7.length_hit].species == "bonobo").sum()), 3)
    r = s7.size_ratio_active_vs_rival
    check("contests decided by more than 20x", int((r > 20).sum()), 34)
    check("chromosomes at or below parity", int((r <= 1).sum()), 3)
    check("CentroSeek recovers the sub-parity set", int(s7[r <= 1].model_hit.sum()), 3)
    check("bonobo chr14 ratio", round(float(r[s7.chrom.str.startswith("chr14_mat")].iloc[0]), 2),
          0.98, 5e-3)
    check("random-pick null, expected ape hits", round((1 / s7.n_arrays).sum(), 1), 18.6, 0.05)

    # ---- Fig. S12, Fig. S13: nulls and resolution ----
    section("Robustness (Fig. S12, Fig. S13)")
    n = T("label_permutation_null.csv")
    check("permutations", len(n), 10000)
    check("null mean AUROC", round(n.perm_auc.mean(), 3), 0.460, 5e-4)
    check("null max AUROC", round(n.perm_auc.max(), 3), 0.798, 5e-4)
    check("null mean top-1", round(n.perm_top1.mean(), 3), 0.295, 5e-4)
    ph = T("window_phase_sensitivity.csv")
    check("arrays in the phase sweep", len(ph), 81)
    check("median phase spread", round(ph.phase_spread.median(), 4), 0.0143, 5e-5)

    # ---- Fig. 3, Table S10 ----
    section("Genome-wide atlas (Fig. 3, Table S10)")
    bc, bcl = T("atlas_by_chrom.csv"), T("atlas_by_class.csv")
    check("chromosomes in the atlas", len(bc), 24)
    check("10-kbp windows", int(bc.n_windows.sum()), 311715)
    low = bcl.nsmallest(2, "bz2_bpb")["class"].tolist()
    check("two most compressible classes", sorted(low), sorted(["alpha_hor", "hsat"]))

    print(f"\n{len(PASS)} claims verified, {len(FAIL)} failed")
    if FAIL:
        for c, got, want in FAIL:
            print(f"  - {c}: got {got}, manuscript says {want}")
        sys.exit(1)


if __name__ == "__main__":
    main()
