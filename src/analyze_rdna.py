"""Does the centromere principle generalize to rDNA? (an honest test)

The centromere result: among α-satellite HOR arrays, the functionally active one
is the most sequence-homogeneous (recently homogenized by concerted evolution).
rDNA arrays also evolve by concerted evolution, and their activity has a
functional readout — methylation (active rDNA is hypomethylated). So the analog
prediction is: the more-active (lower-methylation) rDNA is more homogeneous.

rDNA units are ~45 kb, so 'homogeneity' (unit-to-unit redundancy) only shows at
large window scale; we use 100 kb windows (~2 units). We test within-array and at
the array level (the honest unit), with the CHM13 = hydatidiform-mole caveat noted.

Output: results/tables/rdna_test.csv ; results/figures/figS03_rdna_test.png ; results/rdna_log.txt
"""
import os
import time
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.stats import spearmanr

import sys
sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import bz2_bits_per_base, clean  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
B = "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/"
METH = B + "assemblies/annotation/chm13v2.0_nanopore_CpG.bw"
WIN = 100_000
# (acc, chrom, rDNA start, rDNA end) from CenSat
RDNA = [
    ("NC_060937.1", "chr13", 5_770_548, 9_348_041),
    ("NC_060938.1", "chr14", 2_099_537, 2_817_811),
    ("NC_060939.1", "chr15", 2_506_442, 4_707_485),
    ("NC_060945.1", "chr21", 3_108_298, 5_612_715),
    ("NC_060946.1", "chr22", 4_793_794, 5_720_650),
]
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def fetch(acc, lo, hi, chunk=500_000):
    out = []
    s = lo
    while s <= hi:
        e = min(s + chunk - 1, hi)
        url = EFETCH + "?" + urllib.parse.urlencode(dict(
            db="nuccore", id=acc, rettype="fasta", retmode="text",
            seq_start=s, seq_stop=e))
        req = urllib.request.Request(url, headers={"User-Agent": "GC/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.append("".join(l.strip() for l in r.read().decode().splitlines()
                               if not l.startswith(">")))
        time.sleep(0.34)
        s = e + 1
    return "".join(out)


def main():
    meth = pyBigWig.open(METH)
    rows = []
    for acc, ch, lo, hi in RDNA:
        log(f"[{ch}] rDNA {lo:,}-{hi:,} ...")
        seq = fetch(acc, lo, hi)
        for w0 in range(0, len(seq) - WIN + 1, WIN):
            cseq = clean(seq[w0:w0 + WIN])
            if len(cseq) < 0.5 * WIN:
                continue
            gpos0 = lo + w0
            m = meth.stats(ch, gpos0 - 1, gpos0 + WIN - 1, type="mean")[0]
            rows.append({"chrom": ch, "start": gpos0,
                         "bz2": bz2_bits_per_base(cseq),
                         "meth": m if m is not None else np.nan})
    meth.close()
    df = pd.DataFrame(rows).dropna(subset=["meth"])
    df.to_csv(os.path.join(TAB, "rdna_test.csv"), index=False)
    log(f"\nrDNA 100 kb windows: {len(df)} across {df.chrom.nunique()} arrays")
    log(f"rDNA compressibility: mean bzip2 {df.bz2.mean():.3f} bits/base "
        f"(vs genome non-satellite ~2.21, centromeric HOR ~0.73) — rDNA is "
        f"{'highly compressible' if df.bz2.mean()<1.5 else 'moderately compressible'}.")

    # array-level: homogeneity vs methylation (active = low meth)
    arr = df.groupby("chrom").agg(bz2=("bz2", "mean"), meth=("meth", "mean"),
                                  n=("bz2", "size")).reset_index()
    log("\n== Array-level: homogeneity vs activity (methylation) ==")
    log(arr.round(3).to_string(index=False))
    rho_a, p_a = spearmanr(arr.bz2, arr.meth)
    log(f"\nArray-level Spearman(bzip2, methylation) = {rho_a:+.2f} (p={p_a:.2f}, n=5)")
    log("  positive ρ ⇒ more methylated (silent) arrays are LESS compressible (less "
        "homogeneous), i.e. active(low-meth) = more homogeneous — the centromere parallel.")

    # within-array (avoids between-array baseline confound)
    log("\n== Within-array Spearman(bzip2, methylation) ==")
    within = []
    for ch, g in df.groupby("chrom"):
        if len(g) >= 8:
            r, p = spearmanr(g.bz2, g.meth)
            within.append(r)
            log(f"  {ch}: ρ={r:+.2f} (n={len(g)})")
    log(f"  median within-array ρ = {np.median(within):+.2f}" if within else "  (too few)")

    # verdict
    consistent = (rho_a > 0.2) and (np.median(within) > 0.1 if within else False)
    log("\nVERDICT: " + (
        "the rDNA data show the SAME direction as centromeres (active/low-methylation "
        "rDNA is more homogeneous) — the principle tentatively generalizes."
        if consistent else
        "the rDNA signal is weak/inconsistent — with only 5 arrays (and CHM13 being a "
        "hydatidiform mole with atypical rDNA methylation), this is an underpowered, "
        "inconclusive test, not support for the parallel."))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for ch, g in df.groupby("chrom"):
        ax[0].scatter(g.bz2, g.meth, s=18, alpha=0.6, label=ch)
    ax[0].set_xlabel("bzip2 bits/base (homogeneity →)")
    ax[0].set_ylabel("CpG methylation (silent →)")
    ax[0].set_title(f"rDNA 100 kb windows (within-array median ρ="
                    f"{np.median(within):+.2f})" if within else "rDNA windows")
    ax[0].legend(fontsize=7)
    ax[1].scatter(arr.bz2, arr.meth, s=80, color="#d62728")
    for _, r in arr.iterrows():
        ax[1].annotate(r.chrom, (r.bz2, r.meth), fontsize=8)
    ax[1].set_xlabel("array mean bzip2 (homogeneity →)")
    ax[1].set_ylabel("array mean methylation")
    ax[1].set_title(f"rDNA array level (n=5; ρ={rho_a:+.2f})")
    fig.suptitle("Does the centromere homogeneity principle generalize to rDNA?")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS03_rdna_test.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(ROOT, "results", "rdna_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
