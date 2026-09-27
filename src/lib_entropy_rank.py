"""Entropy-Rank Ratio (R) after Pastore et al., arXiv:2511.05300 (Nov 2025).

Definition. For a target sequence of length L over the DNA alphabet (a=4), let
H_target be its Shannon entropy at order k (entropy of the k-mer frequency
distribution). The Entropy-Rank Ratio is the probability that a random sequence
of the same length L has entropy <= H_target:

    R(seq; k, null) = P_{S ~ null(L)}[ H_k(S) <= H_k(seq) ]

R lies in [0, 1], is distribution-aware, and does NOT saturate the way windowed
Shannon entropy does: two windows that both push H_k to its ceiling can still be
separated by where they sit in the null entropy distribution.

Why k matters. At k=1, H depends only on base composition, so a periodic repeat
(ACGTACGT...) and a random sequence both have H1 = 2 bits and are indistinguish-
able. Structure lives in higher-order statistics, so the analysis uses k >= 2;
the repeat then has few distinct k-mers -> low H_k -> low R, while random
sequence has high H_k -> high R.

Two null models (both reported):
  * "uniform"      : S ~ iid Uniform{A,C,G,T}. Canonical R.
  * "composition"  : S ~ iid with the *target's* base frequencies. Controls for
                     compositional skew, so a low R under this null reflects
                     genuine higher-order repetition, not GC bias.

Exact k=1 case. At k=1 the composition count vector fully determines H, so R is
computed exactly by summing multinomial weights over all 4-part compositions of
L with entropy <= H_target (O(L^3), no sampling). Used to validate the estimator.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

import numpy as np

A = 4
_BASES = np.array(list("ACGT"))
_LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# Core entropy
# --------------------------------------------------------------------------- #
def _clean(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in "ACGT")


def kmer_entropy(seq: str, k: int = 1) -> float:
    """Shannon entropy (bits) of the k-mer frequency distribution."""
    if len(seq) < k:
        return float("nan")
    counts = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return h / _LOG2


def _comp_entropy_from_counts(counts) -> float:
    L = sum(counts)
    if L == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / L
            h -= p * math.log(p)
    return h / _LOG2


def composition_entropy(seq: str) -> float:
    s = _clean(seq)
    return _comp_entropy_from_counts([s.count(b) for b in "ACGT"])


# --------------------------------------------------------------------------- #
# Exact R at k=1 (validation)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _lgamma(n: int) -> float:
    return math.lgamma(n + 1)


def _log_multinomial(counts) -> float:
    lg = _lgamma(sum(counts))
    for c in counts:
        lg -= _lgamma(c)
    return lg


def entropy_rank_ratio_exact_k1(L: int, h_target: float, tol: float = 1e-12) -> float:
    """Exact R at k=1: fraction of a^L space with composition entropy <= target.
    Sums multinomial weights over all 4-part compositions of L. O(L^3)."""
    if L <= 0:
        return float("nan")
    log_total = L * math.log(A)
    acc = 0.0
    cap = h_target + tol
    for nA in range(L + 1):
        for nC in range(L - nA + 1):
            for nG in range(L - nA - nC + 1):
                counts = (nA, nC, nG, L - nA - nC - nG)
                if _comp_entropy_from_counts(counts) <= cap:
                    acc += math.exp(_log_multinomial(counts) - log_total)
    return min(1.0, max(0.0, acc))


# --------------------------------------------------------------------------- #
# Monte-Carlo null distribution of H_k (the analysis workhorse)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def _null_entropy_samples(L: int, k: int, null: str, probs_key: tuple,
                          n_mc: int, seed: int) -> tuple:
    """Cached sorted array of H_k for n_mc random length-L sequences.
    probs_key is the base-probability tuple (only used for null='composition')."""
    rng = np.random.default_rng(seed)
    if null == "uniform":
        p = np.array([0.25, 0.25, 0.25, 0.25])
    elif null == "composition":
        p = np.array(probs_key, dtype=float)
        p = p / p.sum()
    else:
        raise ValueError(f"unknown null model: {null}")
    out = np.empty(n_mc, dtype=float)
    idx = rng.choice(A, size=(n_mc, L), p=p)
    for i in range(n_mc):
        s = "".join(_BASES[idx[i]])
        out[i] = kmer_entropy(s, k)
    out.sort()
    return tuple(out.tolist())


def entropy_rank_ratio(seq: str, k: int = 2, null: str = "uniform",
                       n_mc: int = 2000, seed: int = 7) -> float:
    """R for a DNA window: empirical P[H_k(comparison) <= H_k(seq)] under `null`.

    Null models:
      "uniform"      : iid Uniform{ACGT}. Degenerate on real DNA (R->0) — kept
                       to document that failure mode.
      "composition"  : iid with the window's base frequencies. Also degenerate
                       on real DNA because it ignores di+ nucleotide structure.
      "shuffle"      : dinucleotide-preserving shuffles of the window itself.
                       Destroys long-range repetition while holding 1- and
                       2-mer composition fixed, so R at k>=~8 measures genuine
                       higher-order repetition. THIS is the informative regime.

    For iid nulls the null entropy distribution is cached per
    (L, k, null, composition, n_mc, seed). For the shuffle null it is generated
    per call (it is sequence-specific).
    """
    s = _clean(seq)
    L = len(s)
    if L < k:
        return float("nan")
    h = kmer_entropy(s, k)

    if null == "shuffle":
        from lib_metrics import dinucleotide_shuffle  # local import avoids cycle
        rng = np.random.default_rng(seed if seed else (271 + L))
        null_h = np.fromiter(
            (kmer_entropy(dinucleotide_shuffle(s, rng), k) for _ in range(n_mc)),
            dtype=float, count=n_mc,
        )
        return float(np.mean(null_h <= h + 1e-12))

    if null == "composition":
        comp = [s.count(b) for b in "ACGT"]
        tot = sum(comp) or 1
        probs_key = tuple(round(c / tot, 4) for c in comp)
    else:
        probs_key = (0.25, 0.25, 0.25, 0.25)
    samples = np.asarray(_null_entropy_samples(L, k, null, probs_key, n_mc, seed))
    return float(np.searchsorted(samples, h + 1e-12, side="right") / len(samples))
