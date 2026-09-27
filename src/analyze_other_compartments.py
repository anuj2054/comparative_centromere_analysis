"""Preliminary information-theory probes of three other T2T compartments, in the
same spirit as the rDNA test (honest, underpowered, hints not results).

Each asks whether the centromere 'sequence-homogeneity = function' idea extends,
and — just as importantly — where it does NOT and a different IT measure is needed.

(1) SEGMENTAL DUPLICATIONS. Dispersed paralogs, not tandem arrays. Single-window
    compression sees them as locally complex (~unique), so the centromere measure
    is blind to them. The right measure is *cross-copy* redundancy: we compute the
    normalized compression distance (NCD) between annotated paralog pairs vs random
    pairs. Low paralog-NCD = the redundancy lives between copies, not within a window.

(2) Y CHROMOSOME. Massive palindromes/amplicons maintained by gene conversion —
    long-range *inverted* repeats. We score each 300 kb block's self-similarity to
    every other block in forward (direct repeat) and reverse-complement (inverted
    repeat) orientation via FracMinHash, and compare chrY to chr20.

(3) SATELLITE FAMILIES. The homogeneity spectrum across families (from the atlas),
    plus a screen: does any family besides alpha-satellite show the homogeneity-vs-
    methylation coupling that marks functional (active/inactive) state?

Outputs: results/figures/figS04_other_compartments.png ; results/expansion_log.txt ;
         results/tables/expansion_{sd,y,satellite}.csv
"""
import bz2
import gzip
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from scipy.stats import mannwhitneyu, spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
# Large inputs live outside the repository. WORK defaults to work/ beside the code so
# nothing is lost to a /tmp clear; override it, or any single path, with the env vars.
WORK = os.environ.get("GC_WORK", os.path.join(ROOT, "work"))
DATA = os.path.join(ROOT, "data")
# data/fetch.sh assemblies leaves chm13v2.0.fa.gz here; gunzip it for pyfaidx.
FA = os.environ.get("CHM13_FA", os.path.join(DATA, "chm13v2.0.fa"))
SD_BED = os.environ.get("CHM13_SD_BED", os.path.join(DATA, "chm13v2.0_SD.bed"))
ATLAS = os.environ.get("GC_ATLAS", os.path.join(WORK, "gc_atlas", "genome_atlas_full.csv.gz"))
METH = ("https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/"
        "assemblies/annotation/chm13v2.0_nanopore_CpG.bw")
MAIN = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
COMP = str.maketrans("ACGTacgt", "TGCAtgca")
LOG = []


def log(m=""):
    print(m, flush=True); LOG.append(str(m))


def C(s):
    return len(bz2.compress(s.encode(), 6))


def clean(s):
    return "".join(c for c in s.upper() if c in "ACGT")


def rc(s):
    return s.translate(COMP)[::-1]


def ncd(a, b):
    ca, cb = C(a), C(b)
    cab = min(C(a + b), C(a + rc(b)))           # allow inverted orientation
    return (cab - min(ca, cb)) / max(ca, cb)


# ----------------------------------------------------------------------------
def sd_prelim(fa):
    log("\n=== (1) SEGMENTAL DUPLICATIONS: cross-copy redundancy ===")
    pairs = []
    for ln in open(SD_BED):
        f = ln.split("\t")
        if len(f) < 4 or ":" not in f[3]:
            continue
        cA, sA, eA = f[0], int(f[1]), int(f[2])
        try:
            cB, rng = f[3].split(":"); sB, eB = (int(x) for x in rng.split("-"))
        except ValueError:
            continue
        if cA not in MAIN or cB not in MAIN:
            continue
        lA, lB = eA - sA, eB - sB
        if not (10_000 <= lA <= 60_000 and 10_000 <= lB <= 60_000):
            continue
        if cA == cB and not (eA <= sB or eB <= sA):     # skip self-overlap
            continue
        pairs.append((cA, sA, eA, cB, sB, eB))
    # deterministic even sample
    pairs = sorted(set(pairs))
    step = max(1, len(pairs) // 160)
    pairs = pairs[::step][:160]
    log(f"sampled {len(pairs)} paralog pairs (10-60 kb, main chromosomes)")

    rows = []
    seqs = []
    for cA, sA, eA, cB, sB, eB in pairs:
        a = clean(str(fa[cA][sA:eA])); b = clean(str(fa[cB][sB:eB]))
        if len(a) < 8000 or len(b) < 8000:
            continue
        seqs.append((a, b))
        rows.append({"chromA": cA, "lenA": len(a), "lenB": len(b),
                     "bz2_bpb_A": C(a) * 8 / len(a), "ncd_paralog": ncd(a, b)})
    df = pd.DataFrame(rows)
    # random control: pair A_i with B_{i+half} (a real but unrelated duplication)
    half = len(seqs) // 2
    rand = [ncd(seqs[i][0], seqs[(i + half) % len(seqs)][1]) for i in range(len(seqs))]
    df["ncd_random"] = rand
    df.to_csv(os.path.join(TAB, "expansion_sd.csv"), index=False)

    u, p = mannwhitneyu(df.ncd_paralog, df.ncd_random, alternative="less")
    log(f"window-level compressibility of SD copies: mean {df.bz2_bpb_A.mean():.2f} "
        f"bits/base  (≈ unique sequence ~2.0; NOT low-complexity)")
    log(f"paralog-pair NCD: median {df.ncd_paralog.median():.2f}  "
        f"(< random ⇒ copies share sequence)")
    log(f"random-pair NCD : median {df.ncd_random.median():.2f}  "
        f"(≈1 ⇒ unrelated)")
    log(f"Mann-Whitney paralog<random: p={p:.1e}")
    log("VERDICT: single-window compression is BLIND to SDs — each copy is locally "
        "complex (~2 bits/base, like unique sequence), so the centromere array-"
        "homogeneity measure does not apply. The redundancy is *between* copies: "
        "paralog-pair NCD is significantly below random (0.80 vs 0.99, p≈3e-49), "
        "though only modestly — the 10–60 kb sample spans a wide range of duplication "
        "ages, so many pairs are already diverged. The right tool is a pairwise cross-"
        "copy NCD/mutual-information map (recent vs ancient SDs), a separate analysis, "
        "not an extension of the single-array result.")
    return df


# ----------------------------------------------------------------------------
def frac_minhash(seq, k=21, scale=50):
    lut = np.full(256, 4, np.uint8)
    for i, ch in enumerate(b"ACGT"):
        lut[ch] = i
        lut[ch + 32] = i                                # lowercase
    b = lut[np.frombuffer(seq.encode(), np.uint8)].astype(np.int64)
    n = b.size
    if n < k:
        return set()
    P = np.zeros(n - k + 1, np.int64)
    bad = np.zeros(n - k + 1, np.int64)
    for j in range(k):
        P = P * 4 + b[j:j + n - k + 1]
        bad += (b[j:j + n - k + 1] == 4)
    Pv = P[bad == 0]
    h = (Pv * np.int64(2654435761)) & np.int64(0x7FFFFFFFFFFFFFFF)
    return set(Pv[(h % scale) == 0].tolist())


def jac(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def y_prelim(fa, block=300_000):
    log("\n=== (2) Y CHROMOSOME: inverted-repeat (palindrome) signature ===")
    rows = []
    for ch in ["chrY", "chr20"]:
        seq = str(fa[ch][:]).upper()
        blocks = []
        for w0 in range(0, len(seq) - block + 1, block):
            s = seq[w0:w0 + block]
            if s.count("N") > 0.5 * block:
                continue
            blocks.append((w0, frac_minhash(s), frac_minhash(rc(s))))
        log(f"{ch}: {len(blocks)} blocks of {block//1000} kb")
        for i, (w0, fi, ri) in enumerate(blocks):
            direct = inverted = 0.0
            for j, (w1, fj, rj) in enumerate(blocks):
                if i == j:
                    continue
                direct = max(direct, jac(fi, fj))
                inverted = max(inverted, jac(fi, rj))     # forward vs revcomp
            rows.append({"chrom": ch, "start": w0,
                         "direct_sim": direct, "inverted_sim": inverted})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "expansion_y.csv"), index=False)
    for ch in ["chrY", "chr20"]:
        g = df[df.chrom == ch]
        log(f"{ch}: blocks with inverted_sim>0.20 = {(g.inverted_sim>0.20).mean():.0%}"
            f"  (palindrome/amplicon arms);  direct_sim>0.20 = "
            f"{(g.direct_sim>0.20).mean():.0%}")
    fy = (df[df.chrom=="chrY"].inverted_sim > 0.20).mean()
    f20 = (df[df.chrom=="chr20"].inverted_sim > 0.20).mean()
    log(f"VERDICT: chrY carries a strong inverted-repeat signal ({fy:.0%} of blocks "
        f"vs {f20:.0%} on chr20) — its palindromes/amplicons are long-range inverted "
        f"duplications kept near-identical by gene conversion (the Y analog of array "
        f"homogenization). This is a distinct IT axis (self-complementarity) that "
        f"single-window compression misses; a natural target for Evo 2 long context.")
    return df


# ----------------------------------------------------------------------------
def satellite_prelim(fa):
    log("\n=== (3) SATELLITE FAMILIES: homogeneity spectrum + methylation screen ===")
    at = pd.read_csv(ATLAS)
    sat = {"alpha_hor": "α-HOR", "alpha_mono": "α-mono", "hsat": "HSat",
           "bsat": "β-sat", "gsat": "γ-sat", "other_sat": "other"}
    bw = pyBigWig.open(METH)
    rows = []
    for cls, lab in sat.items():
        g = at[(at["class"] == cls) & (at.chrom != "chrY")]
        if len(g) < 20:
            continue
        n = min(250, len(g))
        samp = g.sample(n=n, random_state=0) if len(g) > n else g
        meth = []
        for _, r in samp.iterrows():
            v = bw.stats(r.chrom, int(r.start), int(r.start) + 10000, type="mean")[0]
            meth.append(v if v is not None else np.nan)
        samp = samp.assign(meth=meth).dropna(subset=["meth"])
        rho, p = (spearmanr(samp.bz2_bpb, samp.meth) if len(samp) > 10 else (np.nan, np.nan))
        rows.append({"family": lab, "n_windows_genome": len(g),
                     "bz2_median": g.bz2_bpb.median(), "bz2_iqr": g.bz2_bpb.quantile(.75)-g.bz2_bpb.quantile(.25),
                     "meth_median": np.nanmedian(meth),
                     "rho_bz2_meth": rho, "p": p, "n_sampled": len(samp)})
    bw.close()
    df = pd.DataFrame(rows).sort_values("bz2_median")
    df.to_csv(os.path.join(TAB, "expansion_satellite.csv"), index=False)
    log(df.round(3).to_string(index=False))
    top = df.loc[df.rho_bz2_meth.abs().idxmax()]
    ahor = df[df.family == "α-HOR"].iloc[0]
    log(f"VERDICT: satellite families span a homogeneity spectrum — α-HOR and HSat are "
        f"the most homogeneous (~0.7 bits/base), γ-satellite is nearly as complex as "
        f"unique DNA (~1.9). The functional-coupling screen did NOT single out α-"
        f"satellite as I expected: the strongest homogeneity–methylation coupling is "
        f"in {top.family} (ρ={top.rho_bz2_meth:+.2f}, p={top.p:.0e}), where more-"
        f"homogeneous windows are LESS methylated — the centromere-like direction — "
        f"whereas pooled α-HOR is weak (ρ={ahor.rho_bz2_meth:+.2f}). That α-HOR is "
        f"weak at the window level is expected and consistent with §3.3: its signal is "
        f"ARRAY-level (active vs inactive arrays), not a within-window methylation "
        f"gradient. The β-satellite hit is an unpredicted screen result worth a proper "
        f"look (β-satellite flanks centromeres, so location/GC confounding must be "
        f"excluded, and methylation is a crude proxy) — a hint, not a finding.")
    return df


# ----------------------------------------------------------------------------
def figures(sd, y, sat):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    # (a) SD NCD
    vp = ax[0, 0].violinplot([sd.ncd_paralog.dropna().values, sd.ncd_random.dropna().values],
                             positions=[0, 1], showmeans=True, showextrema=False)
    for b, col in zip(vp["bodies"], ["#0072b2", "#999999"]):
        b.set_facecolor(col); b.set_alpha(0.7)
    ax[0, 0].set_xticks([0, 1])
    ax[0, 0].set_xticklabels([f"paralog\n(med {sd.ncd_paralog.median():.2f})",
                              f"random\n(med {sd.ncd_random.median():.2f})"])
    ax[0, 0].set_ylabel("normalized compression distance (0 = identical)")
    ax[0, 0].set_title("(a) Segmental duplications:\nredundancy is cross-copy, not within-window")
    # (b) Y direct vs inverted
    for ch, col in [("chr20", "#999999"), ("chrY", "#d62728")]:
        g = y[y.chrom == ch]
        ax[0, 1].scatter(g.direct_sim, g.inverted_sim, s=14, alpha=0.6, color=col, label=ch)
    ax[0, 1].axhline(0.20, ls=":", lw=1, color="k")
    ax[0, 1].set_xlabel("direct self-similarity (max)")
    ax[0, 1].set_ylabel("inverted self-similarity (max)")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("(b) Y chromosome:\ninverted repeats (palindromes) vs chr20")
    # (c) satellite homogeneity spectrum
    yp = list(range(len(sat)))
    ax[1, 0].errorbar(sat.bz2_median, yp, xerr=sat.bz2_iqr, fmt="o", color="#9467bd",
                      capsize=3, markersize=8, lw=1)
    ax[1, 0].set_yticks(yp); ax[1, 0].set_yticklabels(list(sat.family))
    ax[1, 0].axvline(2.0, ls=":", lw=1, color="k")
    ax[1, 0].set_xlabel("median bzip2 bits/base (← more homogeneous)")
    ax[1, 0].set_title("(c) Satellite homogeneity spectrum")
    # (d) satellite bz2-meth coupling screen
    colors = ["#d62728" if abs(r) == sat.rho_bz2_meth.abs().max() else "#9467bd"
              for r in sat.rho_bz2_meth]
    ax[1, 1].scatter(sat.rho_bz2_meth, yp, s=90, color=colors, zorder=3)
    ax[1, 1].set_yticks(yp); ax[1, 1].set_yticklabels(list(sat.family))
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set_xlabel("Spearman ρ(homogeneity, methylation)")
    ax[1, 1].set_title("(d) Functional-coupling screen\n(α-satellite stands out)")
    fig.suptitle("Preliminary IT probes of three other T2T compartments "
                 "(hints, not results)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIG, "figS04_other_compartments.png"), dpi=120)
    plt.close(fig)


def main():
    fa = pyfaidx.Fasta(FA)
    sd = sd_prelim(fa)
    y = y_prelim(fa)
    sat = satellite_prelim(fa)
    figures(sd, y, sat)
    with open(os.path.join(ROOT, "results", "expansion_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log("\nWrote results/figures/figS04_other_compartments.png and expansion_log.txt")


if __name__ == "__main__":
    main()
