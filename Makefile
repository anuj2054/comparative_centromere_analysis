PY ?= python3

.PHONY: help verify model extracts figures figures-raw test clean-pyc

help:
	@echo "make verify       re-derive the manuscript's numbers from results/tables/"
	@echo "make model        refit CentroSeek, check it matches the shipped constants"
	@echo "make extracts     fetch the four small chr21 FASTAs (~5 MB, NCBI)"
	@echo "make figures      13 of 15 figures; fetches the extracts, streams 3 tracks"
	@echo "make figures-raw  Figs. 3 and S4; needs the assembly (see data/fetch.sh)"
	@echo "make test         verify + model, the pre-release gate"

verify:
	$(PY) verify/verify_claims.py

model:
	$(PY) -m centroseek.fit_model

extracts:
	./data/fetch.sh extracts

# Most of these read only results/tables/ and results/evo2/. Three do not, so this
# target is not offline:
#   build_chr21_map.py     recomputes from the data/*.fasta extracts, hence `extracts`
#   analyze_chimp_windows  streams the mPanTro3 assembly and a remote CenSat file
#   analyze_rdna           streams a methylation track and fetches rDNA from NCBI
# The last two rebuild tables that are already deposited. analyze_cdr.py shows the
# guarded pattern: use the deposited table, rebuild only when it is missing.
figures: extracts
	$(PY) src/plot_predictor.py
	$(PY) src/plot_cross_species.py
	$(PY) src/plot_region_classes.py
	$(PY) src/plot_evo2_axis.py
	$(PY) src/plot_permutation_null.py
	$(PY) src/plot_window_phase.py
	$(PY) src/analyze_censat_classes.py
	$(PY) src/analyze_generalization.py
	$(PY) src/analyze_cdr.py
	$(PY) src/analyze_chimp_windows.py
	$(PY) src/analyze_rdna.py
	$(PY) src/build_chr21_map.py        # slow: recomputes the chr21 map, a few minutes
	$(PY) src/analyze_evo2_chr21.py      # run after build_chr21_map

# Fig. 3 recomputes the genome-wide 311,715-window atlas from the assembly.
# Fig. S4 additionally needs pyfaidx, which is not in the published environment.
figures-raw:
	$(PY) src/build_genome_atlas.py
	$(PY) src/analyze_other_compartments.py

test: verify model

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
