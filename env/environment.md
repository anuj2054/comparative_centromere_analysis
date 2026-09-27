# Environment

```
python -m venv venv && source venv/bin/activate
pip install -r env/requirements.txt
```

Python 3.9.6 (CPython), macOS arm64. Exact versions are pinned in
`requirements.txt`.

## Why the pins are tighter than usual

**bzip2 bits/base is not a mathematical quantity.** It is the output size of a
particular compressor on a particular input, so it depends on the implementation.
The values here come from CPython's `bz2` module at `compresslevel=9`, which wraps
libbz2. A different libbz2 build may give marginally different compressed sizes and
therefore marginally different feature values. Nothing in the paper's conclusions
turns on the fourth decimal, but exact reproduction of the deposited tables does.

This matters more than it looks, because array-mean bzip2 already has a measured
resolution floor: shifting the 2 kbp window grid in 500 bp steps moves an array's
own mean by a median of 0.0143 bits/base (Fig. S13). Differences smaller than that
are not interpretable, which is why human chromosome 3, whose two leading candidates
differ by 0.0020, is reported as unresolved by homogeneity rather than as a clean
call.

**Window geometry is part of the feature definition.** 2 kbp windows at 4 kbp
spacing, windows kept when at least half their bases are unambiguous ACGT. Values
computed under a different geometry are not comparable to these.

## Not required for the main results

`pyBigWig` and `pysam` are needed only to read the signal tracks and alignments,
that is, to rebuild the functional labels from scratch. Ranking candidate arrays
with `centroseek` needs neither them nor the tracks.

`pyfaidx` is required by `src/analyze_other_compartments.py` alone, which draws Fig. S4, the
preliminary compartment probes of Supplementary Note 2. It is deliberately
unpinned: it is not installed in the environment that produced the published
numbers, so there is no honest version to pin, and Fig. S4 is the single figure
this environment cannot regenerate. Nothing in the main results depends on it.

**Evo 2** is a separate matter. Per-base surprise was computed with the published
`evo2_7b` checkpoint on GPU, and it is not in `requirements.txt` because it cannot
run in this environment. Evo 2 was never part of the deployed CentroSeek model: it
was not computed for all predictor chromosomes, and chromosome X is excluded from
the Evo 2 analyses entirely because its 1,000 windows were scored against a
mis-fetched chromosome Y record and were withdrawn rather than silently kept. Every
other chromosome X measurement was recomputed from the correct sequence.
