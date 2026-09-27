"""Sequence-complexity metrics: Shannon entropy, Lempel-Ziv, compression bits/base.

All functions operate on uppercase DNA strings over {A,C,G,T}. Non-ACGT
characters (N, etc.) are handled explicitly where noted.
"""
from __future__ import annotations

import bz2
import gzip
import math
from collections import Counter

import numpy as np

ALPHABET = ("A", "C", "G", "T")


def clean(seq: str) -> str:
    """Uppercase and drop non-ACGT characters."""
    seq = seq.upper()
    return "".join(c for c in seq if c in ALPHABET)


def gc_content(seq: str) -> float:
    seq = clean(seq)
    if not seq:
        return float("nan")
    return (seq.count("G") + seq.count("C")) / len(seq)


def shannon_entropy(seq: str, k: int = 1) -> float:
    """Shannon entropy (bits) of the k-mer frequency distribution.

    For k=1 this is the classic per-base entropy, bounded in [0, 2].
    For k>1 it is normalized to bits *per k-mer*; divide by k for bits/base.
    Returns NaN for sequences too short to contain a k-mer.
    """
    seq = clean(seq)
    if len(seq) < k:
        return float("nan")
    counts = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def lz76_complexity(seq: str) -> int:
    """Lempel-Ziv (1976) complexity: number of distinct factors in the
    sequential factorization. A standard algorithmic-complexity proxy.
    """
    return _lz76_kaspar(clean(seq))


def _lz76_kaspar(seq: str) -> int:
    """Canonical Kaspar & Schuster LZ76 complexity counter.

    Tracks k_max (longest reproducible match starting anywhere in the parsed
    prefix) so that completed components advance the prefix by k_max, not by
    the last trial length. Random length-n DNA gives c ~ n / log_a(n).
    """
    n = len(seq)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    c = 1
    k_max = 1
    while True:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:          # exhausted the prefix: close a component
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def lz_normalized(seq: str) -> float:
    """LZ76 complexity normalized by the random-sequence expectation
    n / log_a(n), giving a scale-stable value (~1.0 for random DNA).
    """
    seq = clean(seq)
    n = len(seq)
    if n < 2:
        return float("nan")
    a = len(set(seq)) or 1
    if a < 2:
        return 0.0
    norm = n / (math.log(n) / math.log(a))
    return _lz76_kaspar(seq) / norm


def _bits_per_base(seq: str, compressor) -> float:
    seq = clean(seq)
    if not seq:
        return float("nan")
    raw = seq.encode("ascii")
    comp = compressor(raw)
    # subtract a rough fixed header cost is omitted; report raw bits/base
    return (len(comp) * 8) / len(seq)


def gzip_bits_per_base(seq: str) -> float:
    """gzip compressed size in bits per base. Kolmogorov-complexity proxy."""
    return _bits_per_base(seq, lambda b: gzip.compress(b, 9))


def bz2_bits_per_base(seq: str) -> float:
    """bzip2 (BWT-based) compressed size in bits per base. Often tighter on
    repetitive DNA than gzip; complements the LZ proxy."""
    return _bits_per_base(seq, lambda b: bz2.compress(b, 9))


def two_bit_packed_bits_per_base(seq: str) -> float:
    """Trivial 2-bit packing baseline (no modeling) = 2.0 bits/base for any
    ACGT sequence; included as a reference floor for the compressors."""
    seq = clean(seq)
    return 2.0 if seq else float("nan")


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Shuffle preserving dinucleotide composition (Altschul-Erickson style,
    via an Eulerian-path walk on the dinucleotide graph). Used as a control:
    destroys higher-order structure but keeps 1- and 2-mer frequencies.
    """
    seq = clean(seq)
    if len(seq) < 3:
        return seq
    # Build edge lists per starting nucleotide.
    edges: dict[str, list[str]] = {b: [] for b in ALPHABET}
    for a, b in zip(seq[:-1], seq[1:]):
        edges[a].append(b)
    start, last = seq[0], seq[-1]
    # Eulerian path exists; randomize by shuffling out-edges with a valid
    # last-edge-to-`last` constraint (Altschul-Erickson). Simplified robust
    # approach: retry shuffles until the walk consumes all edges.
    targets = {b: list(v) for b, v in edges.items()}
    counts = {b: len(v) for b, v in edges.items()}
    for _ in range(64):
        order = {b: rng.permutation(len(v)).tolist() for b, v in targets.items()}
        ptr = {b: 0 for b in ALPHABET}
        walk = [start]
        cur = start
        ok = True
        remaining = dict(counts)
        total = sum(counts.values())
        for _step in range(total):
            if ptr[cur] >= len(targets[cur]):
                ok = False
                break
            nxt = targets[cur][order[cur][ptr[cur]]]
            ptr[cur] += 1
            walk.append(nxt)
            cur = nxt
        if ok and len(walk) == total + 1:
            return "".join(walk)
    # Fallback: plain shuffle (preserves only 1-mer composition).
    arr = list(seq)
    rng.shuffle(arr)
    return "".join(arr)
