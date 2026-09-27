# Pipeline

This is a **curated subset** of the working repository, not a copy of it. It holds the
49 scripts that produce a published figure, table or quoted statistic, plus the
modules they import. Superseded and exploratory work — earlier 16-chromosome
analyses, abandoned model searches, the long-context Evo 2 side experiment — is not
here. What remains is what the paper rests on.

Most readers never need this page. `make verify` and `make model` run from the deposited
tables alone, and `make figures` fetches the little it needs. This page matters if you
are rebuilding from raw data.

Every script is named for what it does, so the prefix tells you the kind of thing it is:

| Prefix | Does |
|---|---|
| `fetch_` | pulls input from a primary source |
| `build_` | writes a dataset, a table or a signal track that later stages consume |
| `analyze_` | answers one question and writes the table behind it |
| `check_` | a control or robustness test whose job is to try to break a result |
| `plot_` | draws a published figure and computes nothing new |
| `lib_` | imported by other scripts, never run on its own |

Every file in `src/` carries one of these. `build_bigwig_from_bam.py` is the only one
that nothing else in the repository runs directly: `build_cenpa_track.sh` invokes it as
a subprocess when deeptools will not install.

Figure numbers are deliberately not in filenames. Several scripts draw one figure and
deposit five tables, so naming them for the figure would misdescribe them.
`docs/FIGURES.md` carries the figure-to-script map.

## Stage 0 — acquire

`data/fetch.sh` is the entry point. Subcommands may be combined, so
`./data/fetch.sh assemblies annotations` runs both in that order, and no argument runs
everything.

| Script | Does |
|---|---|
| `data/fetch.sh assemblies` | the four assemblies, from the T2T S3 bucket and GenomeArk |
| `data/fetch.sh annotations` | CenSat, then the exact filenames of the browser-hub tracks to place by hand |
| `data/fetch.sh extracts` | the four small chr21 FASTAs; delegates to `fetch_chr21_extracts.py` and skips files already present unless `FORCE=1` |
| `data/fetch.sh reads` | the ten SRA runs, about 200 GB |
| `fetch_chr21_extracts.py` | cuts three chr21 regions and twelve RefSeq CDS from NCBI and writes `data/manifest.json`, which `build_chr21_map.py` reads the region offset back out of. Standard library only, so it runs before `pip install` |
| `fetch_tracks.py` | CHM13 signal tracks |
| `build_bigwig_from_bam.py` | an aligned BAM to a binned, CPM-normalised bigWig. A pure-Python stand-in for deeptools `bamCoverage`, which does not always install on Apple silicon. The alignment itself, the `bwa mem -k 50 -c 1000000` recipe of Methods 2.4, is in `build_cenpa_track.sh` |

`extracts` is a prerequisite of `make figures`, which runs it for you. It is the one
fetch that costs seconds rather than hours, and
`shasum -a 256 --ignore-missing -c data/checksums.sha256` pins what comes back.

## Where large intermediates go

The genome atlas, the RepeatMasker bed and the sequence caches are too large to commit
and are written to `work/` beside the code, not to `/tmp`, so they survive a reboot.
`build_genome_atlas.py` writes the atlas and both `build_atlas_repeats.py` and
`analyze_other_compartments.py` read it back, so that chain only works if the location
persists. The directory is git-ignored. Override any of it:

| Variable | Default | Used by |
|---|---|---|
| `GC_WORK` | `work/` | the root for everything below |
| `GC_ATLAS_DIR` / `GC_ATLAS` | `work/gc_atlas/` | `build_genome_atlas.py`, `build_atlas_repeats.py`, `analyze_other_compartments.py` |
| `GC_REPEATMASKER` | `work/gc_rm/rm.bed` | `build_atlas_repeats.py` |
| `CHM13_FA`, `CHM13_SD_BED` | `data/` | `analyze_other_compartments.py` (Fig. S4) |
| `SEQ_CACHE`, `GC_SCALE_CACHE` | `work/` | `check_shuffle_control.py`, `check_window_scale.py` |
| `CENPA_WORK`, `CENPA_VENV` | `work/cenpa/` | the five shell scripts |

`data/fetch.sh assemblies` leaves `chm13v2.0.fa.gz` in `data/`; Fig. S4 needs it
gunzipped, since `pyfaidx` cannot index a plain gzip file.

## Stage 1 — shared modules

`lib_metrics.py` (bzip2, Shannon, LZ), `lib_windows.py` (tiling), `lib_entropy_rank.py`
(Entropy-Rank Ratio and its shuffle null), `lib_plotting.py` (violin/box drawing).

These define the features. `centroseek/features.py` reimplements the two that the
tool needs, standalone and dependency-free; the two agree to four decimals on the
deposited arrays.

## Stage 2 — human candidate arrays and labels

| Script | Produces |
|---|---|
| `build_human_arrays.py` | `array_level_19chrom.csv`, the 81-array analysis set |
| `build_human_arrays_all.py` | all 114 annotated arrays, no size filter |
| `analyze_cenpa_occupancy.py` | CENP-A occupancy per array |
| `check_cenpa_aggregation.py` | the three aggregation schemes of Methods 2.4 |
| `check_track_concordance.py` | agreement among the four CENP-A tracks |

## Stage 3 — predictor comparison

| Script | Produces |
|---|---|
| `analyze_predictor_comparison.py` | the leave-one-chromosome-out engine: length vs homogeneity vs both. Writes `PREDICTOR_comparison.csv` and `PREDICTOR_per_chrom_hits.csv`; `check_permutation_nulls.py` imports it so the null and the estimate come from identical code |
| `analyze_length_vs_homogeneity.py` | the AUROC table behind Fig. 1f |
| `analyze_predictor_transfer.py`, `build_multichrom_windows.py` | generalisation across chromosomes |
| `build_window_features.py` | the window-level feature cache |

## Stage 4 — great apes

| Script | Produces |
|---|---|
| `build_ape_censat_census.py` | eligibility: which chromosomes have both `active_hor` and `dhor` |
| `build_ape_arrays.py` | the 47-chromosome primary-haplotype set |
| `build_ape_arrays_alt.py` | alternate haplotypes; the 37/37, 25/25 and 22/28 replication |
| `analyze_chimp_windows.py`, `analyze_cross_species.py` | per-species summaries, Table S9 |
| `check_bonobo_lineage.py` | Fisher exact on the bonobo concentration |

## Stage 5 — atlas and Evo 2

`build_genome_atlas.py` and `build_atlas_repeats.py` build the 311,715-window atlas.
`build_evo2_scores.py` with `build_evo2_chr21.py` / `build_evo2_multichrom.py` ran the
`evo2_7b` checkpoint on GPU; `analyze_evo2_chr21.py` merges the result. Evo 2 cannot run
in the pinned environment — see `env/environment.md`.

## Stage 6 — robustness

`check_permutation_nulls.py` (label-permutation and random-pick nulls),
`check_window_phase.py` (the 0.0143 resolution floor),
`check_paired_within_chrom.py` (sign and Wilcoxon tests),
`check_window_scale.py` and `check_shuffle_control.py` / `check_repeat_period.py` (window-scale sweep at
1, 2, 5, 10, 20 and 50 kbp, and the dinucleotide-shuffle control).

## The five shell scripts

`src/` holds five bash scripts that build the functional labels from raw reads. They are
long-running, need the toolchain and 150 GB of free disk, and nothing in the `Makefile`
calls them. Start with `build_cenpa_track.sh`, the only one that takes subcommands:
`setup`, `check` (a preflight on tools, disk and RAM), `pilot N` to measure cost on two
million read pairs first, then `run N`. The other four run straight through.

| Script | Does |
|---|---|
| `build_alignment_tools.sh` | compiles `bwa` and `samtools` into `~/.local` from source, and sets up the venv. For machines without admin rights; substitutes `cutadapt` for `fastp`, and explains why |
| `build_cenpa_track.sh` | the main one. Builds a CHM13-native CENP-A track from PRJNA559484 reads, replacing the HG002 cross-individual proxy the first submission used. Falls back to `build_bigwig_from_bam.py` where deeptools will not install |
| `build_igg_track.sh` | the matched IgG CUT&RUN control track, so array signal can be read as CENP-A/IgG ratio rather than raw coverage |
| `build_chipseq_track.sh` | the orthogonal CENP-A ChIP-seq with matched input, a second assay against the CUT&RUN label |
| `build_highsalt_track.sh` | the high-salt CUT&RUN pair, testing whether the calls depend on the salt fraction |

They run in that order, each one adding an assay against the CENP-A label the first
builds. None is needed to reproduce a published figure or table; they exist so the label
itself can be rebuilt rather than taken on trust.

## Stage 7 — deposited tables

```
build_human_arrays_all.py  ->  MASTER_human_all_arrays.csv
build_master_tables.py       ->  MASTER_human_19chrom.csv, MASTER_ape_47chrom.csv
build_supplementary_tables.py ->  TableS5, TableS6, TableS7
```

`build_supplementary_tables.py` re-derives 28 manuscript claims before it writes, and
refuses to deposit a table that disagrees with the paper.

## Stage 8 — figures

See `docs/FIGURES.md`. Thirteen of the fifteen are built by `make figures`, ten of them
from the deposited data alone; that page names the three that reach the network and says
which two need not.
