# Which script makes which figure

All write into `results/figures/`. Thirteen are built by `make figures`; the two that
recompute from the assembly are listed separately (`make figures-raw`).

**`make figures` fetches what it needs.** Ten of its thirteen scripts read only
`results/tables/` and `results/evo2/`. Three go to the primary sources, which is the
same contract as the rest of the repository: start from `data/fetch.sh` and everything
downstream follows.

| Script | Reads |
|---|---|
| `build_chr21_map.py` | the four `data/*.fasta` chr21 extracts, which `make figures` fetches first via `make extracts` |
| `analyze_chimp_windows.py` | the mPanTro3 assembly and a CenSat file, streamed over HTTPS |
| `analyze_rdna.py` | a methylation track, and rDNA sequence from NCBI |

The last two recompute tables that are also deposited, so they need the network even
when the table is present. `analyze_cdr.py` takes the other approach and reads the
deposited table when it exists. Both are fine; neither is offline.

| Figure | Script |
|---|---|
| Fig. 1 | `src/plot_predictor.py` |
| Fig. 2 | `src/plot_cross_species.py` |
| Fig. 3 | `src/build_genome_atlas.py` — **needs the assembly**, recomputes the 311,715-window atlas |
| Fig. S1 | `src/plot_region_classes.py` |
| Fig. S2 | `src/plot_evo2_axis.py` |
| Fig. S3 | `src/analyze_rdna.py` |
| Fig. S4 | `src/analyze_other_compartments.py` — **needs the assembly and `pyfaidx`**; the one figure the pinned environment cannot regenerate |
| Fig. S5 | `src/build_chr21_map.py` — rebuilds from `data/chr21_map.fasta`; slow, a few minutes |
| Fig. S6 | `src/analyze_evo2_chr21.py` — run after `build_chr21_map.py` |
| Fig. S7 | `src/analyze_censat_classes.py` |
| Fig. S8, S9 | `src/analyze_generalization.py` |
| Fig. S10 | `src/analyze_chimp_windows.py` |
| Fig. S11 | `src/analyze_cdr.py` |
| Fig. S12 | `src/plot_permutation_null.py` |
| Fig. S13 | `src/plot_window_phase.py` |

## What is not here

`results/figures/` holds the 16 published figure images and nothing else. Several
scripts also emit intermediate panels and diagnostics as a by-product: the two halves
Fig. S1 was once assembled from by hand, the same for Fig. S2, a CENP-A occupancy plot,
a window-level active/inactive split, a genome-wide transfer check, a functional-signal
correlation plot, and an early cross-species summary that Fig. 2 superseded. None is
cited in the paper, so none is deposited. Run the script and it writes its own.

## Panel notes worth knowing before editing

**Fig. 1e** exists to show that array length and homogeneity are coupled at Spearman
rho = -0.51. It is the concession panel: the two features are not independent signals.

**Fig. 2e** whiskers are Wilson score intervals, not bootstrap. Resampling chromosomes
for a perfect score such as gorilla's 13/13 can only return a perfect score, which would
collapse the interval to a point and read as zero uncertainty. There is deliberately no
chance line: the random-pick expectation follows each chromosome's own candidate count
and differs by species (0.379, 0.442, 0.378, 0.396), so one line would be the null for
none of them. Fig. S12c handles chance properly.

**Fig. 2f** marks CentroSeek's errors on every chromosome, not only on the three it
rescues. Marking only the rescues would show the upside of the trade-off and hide its
cost, which is nine chromosomes at ratios 1.96 to 150 that the length rule gets right.
