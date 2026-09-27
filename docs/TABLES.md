# Deposited tables

A curated subset: 60 tables, being those a published figure, table or quoted
statistic depends on. See `docs/PIPELINE.md` for what was left out and why.

Everything in `results/tables/` that a manuscript object depends on, and what it
supports. Regenerate the master tables with `src/build_human_arrays_all.py` and
`src/build_master_tables.py`.

**Tables S5, S6 and S7 are not deposited under those names.** Each was a byte-identical
copy of a `MASTER_` file, so only the master is kept and the table below says which is
which. `src/build_supplementary_tables.py` writes the `TableS*` filenames from the
masters when you want them, and re-derives 28 manuscript claims before it will.

## Primary

| File | Rows | Supports |
|---|---:|---|
| `MASTER_human_all_arrays.csv` **is Table S5** | 114 | Table S5. Every annotated human HOR array on the 19 testable chromosomes; 81 pass the >=10 kbp / >=5-window threshold |
| `MASTER_human_19chrom.csv` **is Table S6** | 19 | Table 1, Table S6 |
| `MASTER_ape_47chrom.csv` **is Table S7** | 47 | Table 2, Table S7, Fig. 2e, Fig. 2f |
| `array_level_19chrom.csv` | 81 | Results 3.1, Fig. 1d-f; the CentroSeek training table |
| `ape_expanded_arrays.csv` / `ape_expanded_chrom.csv` | | Fig. 2a-d |
| `atlas_by_chrom.csv` / `atlas_by_class.csv` / `atlas_repeat_classes.csv` | 24 / 7 / 10 | Fig. 3, Table S10 |
| `label_permutation_null.csv` | 10,000 | Fig. S12 |
| `window_phase_sensitivity.csv` | 81 | Fig. S13 |
| `cross_species_summary.csv` | 4 | Table S9 |
| `censat_class_means.csv` | 4 | Table S4, Fig. S7 |
| `class_summary.csv` / `class_features.csv` | | Table S1, Fig. S1 |
| `h1_nested_models.csv` / `evo2_nested_models.csv` | | Table S2, Table S3, Fig. S2 |
| `generalization_annotation_auroc.csv` | 7 | Fig. S8, Fig. S9 |
| `cdr_correlations.csv` / `cdr_active_windows.csv` | 6 / 2,093 | Fig. S11 |
| `track_concordance.csv` / `cenpa_margin_by_track.csv` | 19 | Results 3.1, CENP-A track agreement |
| `bonobo_lineage_test.csv` | 2 | Fisher exact test, Results 3.3 |
| `rdna_test.csv` | 98 | Fig. S3 (preliminary) |

## Upstream and control tables

These three back a manuscript claim but are read by no script in this subset, because the
script that consumed them was either upstream of the deposit or is not part of the
published pipeline. They are deposited for provenance.

| File | Rows | Supports |
|---|---:|---|
| `APE_locked_47chrom.csv` | 47 | the locked ape working table that `MASTER_ape_47chrom.csv` is cut from. It carries the columns the master drops: per-chromosome difficulty grade, GC by array class, and the separate length, homogeneity and model calls with their hit flags |
| `array_level_19chrom_with_igg.csv` | 81 | the IgG control: `array_level_19chrom.csv` with `igg` and CENP-A/IgG `ratio` columns added and `cenpa_active` recomputed from the ratio, showing the CENP-A label survives input normalisation. `src/build_igg_track.sh` builds the IgG track it reads; the table itself was written upstream by a script not carried into this subset |
| `ape_array_level_p.csv` | 3 | per-species array-level statistics for chimp, gorilla and bonobo: active and inactive array counts, window-level and array-level p, and array AUROC |

## Two traps

**`cdr_active_windows.csv` is the unfiltered build.** It has 2,093 rows; the Fig. S11
correlations use the 2,059 that retain methylation, CENP-A and bzip2 values. Recomputing
the pooled rho on the file as deposited gives -0.411 rather than the -0.43 in the caption.
Apply the same `dropna` before comparing.

**The superseded 16-chromosome tables are not here.** `array_level_stats.csv` and the
other pre-expansion intermediates were dropped when this subset was cut. That file held a
14/16 agreement, 11/16, LOCO AUROC 0.758 and an array-level p of 1.1e-7, none of it cited
in the manuscript, and its p-value is easy to mistake for the current CenSat-labelled
array test, which is 27 active against 54 inactive at p = 9.7e-9 over the 81-array set.
During revision that exact confusion put the stale value into a supplementary table. The
working repository retains them for provenance; this one does not, so the mistake cannot
be repeated here.
