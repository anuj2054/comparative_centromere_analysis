# CentroSeek

Which α-satellite higher-order-repeat array on a chromosome is the functional
centromere, decided from assembled sequence alone.

This repository holds the software and the analysis behind *Active Centromeres Share
a Conserved Array-Architecture Signature Across Great Ape Genomes*. It does two
separable things: it ships a small tool you can point at a new assembly, and it
reproduces every number and figure in the paper.

## The result, stated honestly

Active centromeric arrays are both **expanded** and **more sequence-homogeneous**
than the relict arrays they compete with. Of those two, expansion carries almost all
of the usable signal:

| Rule | Human (19 chr) | Great apes (47 chr) |
|---|---:|---:|
| **Longest array** | **19 / 19** | **44 / 47** |
| CentroSeek (bzip2 + H₁₁) | 17 / 19 | 38 / 47 |
| Most homogeneous array | 17 / 19 | 35 / 47 |

**The one-line rule beats the model.** We are not burying that. CentroSeek earns its
place for a narrower reason: the three chromosomes where the length rule fails are all
bonobo, and CentroSeek recovers all three. It is an exception-finder, not a better
general predictor, and the two should be run together. Where they disagree is where
the biology is interesting.

Length and homogeneity are also strongly coupled (Spearman ρ = −0.51), so they are not
independent evidence. Adding homogeneity to length changes AUROC by −0.004 (95% CI
−0.032 to +0.020).

## Use it on your own assembly

```bash
pip install -r env/requirements.txt
python -m centroseek.rank --fasta genome.fa --arrays candidates.bed --out ranked.tsv
```

`candidates.bed` lists the α-satellite arrays to compare: `chrom start end [name]`.
Output gives each array's bzip2 bits/base, H₁₁, CentroSeek probability, and its rank
under both CentroSeek and the longest-array rule, ranked **within each chromosome**.

### What it does not do

- **It ranks; it does not discover.** Candidate arrays must already be delimited, by
  CenSat or equivalent. A centromere formed outside α-satellite sequence never enters
  the candidate set, so neocentromeres such as the orangutan chromosome 10 case are
  invisible to it by construction.
- **The score is within-chromosome only.** It is uncalibrated across chromosomes and
  meaningless genome-wide.
- **One array per chromosome.** It cannot represent the di- and tri-kinetochore
  centromeres reported at population scale, nor locate the CENP-A domain inside a
  large active array.
- **It is not a substitute for CENP-A or methylation data.** It is a first-pass
  ranking for assemblies that have neither.

## Reproduce the paper

```bash
make verify     # 41 manuscript numbers, re-derived from the deposited tables
make model      # refit CentroSeek and check it matches the shipped constants
make figures    # 13 of the 15 figures
```

`make verify` and `make model` need nothing but the repository. `make figures` is not
offline: it first fetches the four chr21 extracts, 5 MB from NCBI, and three of its
scripts stream remote tracks rather than read the deposited table they would rebuild.
`docs/FIGURES.md` names them.

`make figures-raw` rebuilds the remaining two, which recompute from the assembly:
Fig. 3's 311,715-window genome atlas and Fig. S4.

`verify_claims.py` recomputes 41 published numbers from the deposited tables and exits
non-zero if any has drifted. It exists because drift is not hypothetical: during
revision a stale p-value from a superseded 16-chromosome analysis reached a
supplementary table, and a ratio column stored at one decimal was printed at two. Both
would have failed here.

Rebuilding from raw data needs the assemblies and reads:

```bash
./data/fetch.sh assemblies annotations    # ~4 GB, primary sources, nothing redistributed here
./data/fetch.sh extracts                  # ~5 MB of chr21 sequence, from NCBI
shasum -a 256 -c data/checksums.sha256
```

Nothing under `data/` is redistributed, with no exceptions. Everything there is a fetch
script, a manifest of the regions cut, or the hashes that pin what comes back.

## Layout

```
centroseek/     the tool: two features, five constants, a CLI
src/            the 49 scripts behind published results (see docs/PIPELINE.md)
results/tables/ the 60 tables a published figure, table or statistic depends on
results/evo2/   Evo 2 per-base surprise, precomputed on GPU
results/figures/ the 16 published figure images, and nothing else
verify/         re-derives the manuscript's numbers
data/           fetch scripts and checksums; no redistributed data
docs/           TABLES.md, FIGURES.md
env/            pinned versions, and why the pins are tight
```

This is a **curated subset of the working repository, not a copy of it**: the scripts and
tables kept are those a published figure, table or quoted statistic depends on. Superseded
16-chromosome analyses, abandoned model searches and side experiments are not here.
`docs/PIPELINE.md` says what was cut and why; `docs/TABLES.md` maps each table to what it
supports and flags the one deposited file that is a pre-filter build.

## The model

Logistic regression on two features, both array-level means over 2 kbp windows at 4 kbp
spacing: bzip2 bits/base and Shannon H₁₁. The constants are in `centroseek/model.json`
with their provenance; `python -m centroseek.fit_model` refits them from
`results/tables/array_level_19chrom.csv` and checks they still match.

One provenance detail is recorded there rather than hidden: the shipped coefficients are
fit on the **CenSat** `active_hor` label, while the manuscript **scores** the predictor
against the **CENP-A**-defined label. The two agree on which array ranks top on all 19
chromosomes but differ at the array level, 27 CenSat-active against 19 CENP-A-active, so
they do not give identical coefficients. `--label cenpa` refits on the other one.

Window geometry is part of the feature definition. Array-mean bzip2 moves by a median of
0.0143 bits/base when the grid origin shifts, so values computed under a different
geometry are not comparable — see `env/environment.md`.

## Data sources

T2T-CHM13 v2.0 (GCA_009914755.4); CenSat v2.1; CHM13 Fiber-seq and ONT CpG methylation;
CENP-A CUT&RUN and matched IgG from BioProject PRJNA559484 (low-salt SRR15395851,
SRR15395853; IgG SRR15395849, SRR15395847; high-salt SRR15395852, SRR15395850) and
CENP-A ChIP-seq SRR13278683, SRR13278684 against inputs SRR13278681, SRR13278682;
great-ape assemblies and CenSat from GenomeArk (mPanTro3, mGorGor1, mPanPan1).

Ape active-array calls are graded against each species' own CenSat annotation, which
incorporates methylation and centromere-dip-region information from native long reads.
They are **not** direct CENP-A measurements. The human analysis supplies the CENP-A
validation; the ape analysis tests conservation of the same relationship.

## Licence and citation

Code under `LICENSE-code` (MIT). Derived data tables under `LICENSE-data` (CC BY 4.0).
Neither covers the third-party assemblies, annotations or reads, which remain under
their original terms. See `CITATION.cff`.
