"""The two sequence features CentroSeek uses, computed exactly as in the paper.

Both are array-level means over 2 kbp windows tiled at 4 kbp spacing. The window
geometry matters: array-mean bzip2 shifts by a median of 0.0143 bits/base when the
grid origin moves (Fig. S13), which is larger than the gap between the leading
candidates on human chromosome 3. Do not compare values computed under different
window geometries.

bzip2 bits/base is a property of a particular compressor implementation, not a
mathematical quantity. Values here come from CPython's bz2 module at compresslevel
9. A different libbz2 may give slightly different numbers; see env/environment.md.
"""
import bz2
from collections import Counter
from math import log2

WINDOW = 2000
STEP = 4000
K = 11
MIN_FRACTION = 0.5   # a window needs this fraction of unambiguous bases to count


def clean(seq):
    """Uppercase ACGT only. N and IUPAC ambiguity codes are dropped, not substituted."""
    return "".join(c for c in seq.upper() if c in "ACGT")


def bz2_bits_per_base(seq):
    """Compressed size of the window in bits per base."""
    if not seq:
        return None
    return len(bz2.compress(seq.encode("ascii"), 9)) * 8.0 / len(seq)


def shannon_k(seq, k=K):
    """Shannon entropy of the k-mer distribution, in bits.

    Counts overlapping k-mers. With k=11 and a 2 kbp window most k-mers are unique
    in non-repetitive sequence, so H11 saturates near log2(n_kmers); it separates
    satellite from non-satellite precisely because repeats break that saturation.
    """
    if len(seq) < k:
        return None
    counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    n = sum(counts.values())
    return -sum((c / n) * log2(c / n) for c in counts.values())


def windows(seq, window=WINDOW, step=STEP, offset=0):
    """Tile the sequence, keeping windows with enough unambiguous sequence."""
    for start in range(offset, len(seq) - window + 1, step):
        w = clean(seq[start:start + window])
        if len(w) >= MIN_FRACTION * window:
            yield w


def array_features(seq, window=WINDOW, step=STEP, offset=0):
    """Array-level mean bzip2 bits/base and mean H11.

    Returns (bz2_bpb, H11, n_windows). Returns (None, None, n) when the array is too
    short to yield any usable window; the manuscript excludes such arrays from the
    primary analysis via a >=10 kbp and >=5-window threshold.
    """
    b, h = [], []
    for w in windows(seq, window, step, offset):
        vb, vh = bz2_bits_per_base(w), shannon_k(w)
        if vb is not None:
            b.append(vb)
        if vh is not None:
            h.append(vh)
    if not b:
        return None, None, 0
    return sum(b) / len(b), sum(h) / len(h), len(b)
