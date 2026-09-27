"""Sliding-window engine producing a per-window complexity table."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lib_entropy_rank import entropy_rank_ratio, kmer_entropy
from lib_metrics import bz2_bits_per_base, clean, gc_content, gzip_bits_per_base, lz_normalized


@dataclass
class WindowConfig:
    window: int = 2000
    step: int = 2000
    k_low: int = 1            # classic per-base Shannon entropy (saturates)
    k_mid: int = 3
    k_high: int = 11          # order where higher-order repetition manifests
    r_n_mc: int = 60          # shuffle-null sample size for R
    min_acgt_frac: float = 0.5


def _feature_row(cseq: str, cfg: WindowConfig) -> dict:
    return {
        "n_acgt": len(cseq),
        "gc": gc_content(cseq),
        "H1": kmer_entropy(cseq, cfg.k_low),
        f"H{cfg.k_mid}": kmer_entropy(cseq, cfg.k_mid),
        f"H{cfg.k_high}": kmer_entropy(cseq, cfg.k_high),
        "lz_norm": lz_normalized(cseq),
        "gzip_bpb": gzip_bits_per_base(cseq),
        "bz2_bpb": bz2_bits_per_base(cseq),
        # iid-null R at low k: documents the degenerate regime (-> 0 on real DNA)
        "R_uniform": entropy_rank_ratio(cseq, k=2, null="uniform", n_mc=1500),
        # shuffle-null R at high k: the informative metric
        "R_shuffle": entropy_rank_ratio(cseq, k=cfg.k_high, null="shuffle",
                                        n_mc=cfg.r_n_mc),
    }


def window_metrics(seq: str, cfg: WindowConfig, start_offset: int = 0) -> pd.DataFrame:
    """Sliding-window metrics over `seq`. 0-based coords + start_offset."""
    rows = []
    n = len(seq)
    for s in range(0, max(1, n - cfg.window + 1), cfg.step):
        cseq = clean(seq[s : s + cfg.window])
        if len(cseq) < cfg.min_acgt_frac * cfg.window:
            continue
        row = {"start": start_offset + s, "end": start_offset + s + cfg.window}
        row.update(_feature_row(cseq, cfg))
        rows.append(row)
    return pd.DataFrame(rows)


def fragment_metrics(seq: str, cfg: WindowConfig, label: str,
                     max_frags: int | None = None) -> pd.DataFrame:
    """Non-overlapping fixed-size fragments of `seq`, each a labeled row."""
    rows = []
    cseq = clean(seq)
    n_frag = len(cseq) // cfg.window
    if max_frags:
        n_frag = min(n_frag, max_frags)
    for i in range(n_frag):
        frag = cseq[i * cfg.window : (i + 1) * cfg.window]
        row = {"label": label, "frag": i}
        row.update(_feature_row(frag, cfg))
        rows.append(row)
    return pd.DataFrame(rows)
