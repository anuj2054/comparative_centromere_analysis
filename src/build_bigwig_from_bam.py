"""Fallback for deeptools bamCoverage: BAM -> binned, CPM-normalised bigWig.

deeptools does not always install cleanly on Apple silicon. This needs only pysam
and pyBigWig and reproduces the settings build_cenpa_track.sh asks bamCoverage for:
1 kb bins, CPM normalisation, fragment extension.

Fragment extension is done by counting each properly-paired read-1 across
[pos, pos + template_length) rather than just its own footprint, which is what
--extendReads does for paired data. Unpaired or improperly-paired reads are counted
over their aligned span.

  python build_bigwig_from_bam.py in.bam out.bw [bin_size]
"""
import sys

import numpy as np
import pyBigWig
import pysam

BIN = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
bam_path, bw_path = sys.argv[1], sys.argv[2]

bam = pysam.AlignmentFile(bam_path, "rb")
chroms = [(c, l) for c, l in zip(bam.references, bam.lengths)]
cov = {c: np.zeros(l // BIN + 1, dtype=np.float64) for c, l in chroms}

total = 0
for i, (chrom, length) in enumerate(chroms, 1):
    arr = cov[chrom]
    for r in bam.fetch(chrom):
        if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate:
            continue
        if r.is_paired and r.is_proper_pair:
            if r.is_read2:
                continue                      # count each fragment once
            tlen = abs(r.template_length)
            if not tlen:
                continue
            start = min(r.reference_start, r.next_reference_start)
            end = start + tlen
        else:
            start, end = r.reference_start, r.reference_end or r.reference_start + 1
        total += 1
        b0, b1 = start // BIN, min((end - 1) // BIN, len(arr) - 1)
        if b0 == b1:
            arr[b0] += (end - start) / BIN
        else:
            arr[b0] += ((b0 + 1) * BIN - start) / BIN
            arr[b1] += (end - b1 * BIN) / BIN
            if b1 > b0 + 1:
                arr[b0 + 1:b1] += 1.0
    print(f"  [{i}/{len(chroms)}] {chrom} done", flush=True)
bam.close()

if not total:
    sys.exit("no fragments counted -- is the BAM empty or unindexed?")
scale = 1e6 / total
print(f"fragments counted: {total:,}   CPM scale: {scale:.6g}")

bw = pyBigWig.open(bw_path, "w")
bw.addHeader(chroms)
for chrom, length in chroms:
    arr = cov[chrom] * scale
    nz = np.nonzero(arr)[0]
    if not len(nz):
        continue
    starts = (nz * BIN).astype(int)
    ends = np.minimum(starts + BIN, length).astype(int)
    bw.addEntries([chrom] * len(nz), starts.tolist(),
                  ends=ends.tolist(), values=arr[nz].astype(float).tolist())
bw.close()
print(f"wrote {bw_path}")
