"""Evo 2 "AI-surprise" and embedding extraction (GPU-gated).

This is the long-context foundation-model arm of the proposal. Evo 2 (Arc
Institute, Nature 2026; StripedHyena 2) provides single-nucleotide-resolution
likelihoods over a 1 Mbp context. For each window we extract:

  * per-base negative log-likelihood ("AI-surprise"), mean and profile
  * a pooled embedding from a long-context layer

Evo 2 requires a CUDA GPU (recommended: H100/A100, 40GB+) and the `evo2`
package + model weights. None of that exists on a CPU-only laptop, so this
module is written to:

  1. run for real when a GPU + weights are present (`--backend evo2`), and
  2. otherwise exit cleanly with the exact install/run instructions, OR run a
     deterministic *stub* backend (`--backend stub`) so the surrounding
     pipeline and tests are exercisable without a GPU.

NOTHING here fabricates Evo 2 results. The stub is an explicit, labeled
placeholder (an order-2 Markov surprogate) used only for plumbing tests; its
outputs are written to a separate file and are never presented as Evo 2 output.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #
def evo2_available() -> bool:
    try:
        import torch  # noqa: F401
        import evo2  # noqa: F401
        import torch as _t
        return _t.cuda.is_available()
    except Exception:
        return False


def run_evo2(sequences: list[str], model_name: str = "evo2_7b") -> list[dict]:
    """Real Evo 2 inference. Returns per-sequence mean NLL (bits/base) and a
    pooled embedding. Requires GPU + `evo2` weights."""
    import torch
    from evo2 import Evo2

    model = Evo2(model_name)
    out = []
    for seq in sequences:
        # Evo 2 exposes token logits; per-base NLL = cross-entropy of the
        # next-token prediction over the sequence.
        input_ids = model.tokenizer.tokenize(seq)
        t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
        with torch.no_grad():
            logits, embeddings = model(t, return_embeddings=True,
                                       layer_names=["blocks.28.mlp.l3"])
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = t[0, 1:]
        nll_nats = -logp[torch.arange(len(tgt)), tgt]
        bits_per_base = float(nll_nats.mean().item() / math.log(2))
        emb = embeddings["blocks.28.mlp.l3"][0].float().mean(0).cpu().numpy()
        out.append({"ai_surprise_bpb": bits_per_base, "embedding": emb})
    return out


# --------------------------------------------------------------------------- #
# Stub backend (labeled placeholder; NOT Evo 2)
# --------------------------------------------------------------------------- #
def run_stub(sequences: list[str], order: int = 2) -> list[dict]:
    """Deterministic order-`order` Markov cross-entropy as a stand-in for the
    foundation-model surprise, purely to exercise the pipeline without a GPU.
    Clearly labeled; outputs are written to *_STUB files only."""
    from collections import defaultdict

    out = []
    for seq in sequences:
        s = "".join(c for c in seq.upper() if c in "ACGT")
        if len(s) <= order:
            out.append({"ai_surprise_bpb": float("nan"), "embedding": None})
            continue
        ctx = defaultdict(lambda: defaultdict(int))
        for i in range(len(s) - order):
            ctx[s[i : i + order]][s[i + order]] += 1
        nll = 0.0
        for i in range(len(s) - order):
            d = ctx[s[i : i + order]]
            tot = sum(d.values())
            p = d[s[i + order]] / tot
            nll -= math.log2(p)
        out.append({"ai_surprise_bpb": nll / (len(s) - order),
                    "embedding": None, "backend": "STUB"})
    return out


INSTRUCTIONS = """\
Evo 2 real inference requires a CUDA GPU and the model weights. To run:

    pip install evo2                      # or build from github.com/ArcInstitute/evo2
    # weights auto-download on first use (evo2_7b ~ tens of GB)
    python src/build_evo2_scores.py --backend evo2 --fasta data/satellite.fasta

Recommended hardware: NVIDIA H100/A100 (40GB+). On a CPU-only machine this
backend is unavailable; use --backend stub for a pipeline smoke-test (its
outputs are labeled STUB and must never be reported as Evo 2 results).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["evo2", "stub", "auto"], default="auto")
    ap.add_argument("--fasta", default=os.path.join(os.path.dirname(__file__),
                                                     "..", "data", "satellite.fasta"))
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--max_windows", type=int, default=20)
    args = ap.parse_args()

    seq = "".join(l.strip() for l in open(args.fasta) if not l.startswith(">"))
    seq = "".join(c for c in seq.upper() if c in "ACGT")
    wins = [seq[i * args.window : (i + 1) * args.window]
            for i in range(min(args.max_windows, len(seq) // args.window))]

    backend = args.backend
    if backend == "auto":
        backend = "evo2" if evo2_available() else "stub"

    if backend == "evo2":
        if not evo2_available():
            print("Evo 2 backend unavailable on this machine.\n")
            print(INSTRUCTIONS)
            sys.exit(2)
        res = run_evo2(wins)
        tag = ""
    else:
        print("[Evo 2 unavailable — running STUB backend. Outputs are NOT Evo 2 "
              "results and are written to *_STUB files.]\n")
        print(INSTRUCTIONS)
        res = run_stub(wins)
        tag = "_STUB"

    bpb = np.array([r["ai_surprise_bpb"] for r in res])
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(os.path.join(outdir, f"ai_surprise{tag}.csv"), bpb,
               delimiter=",", header="ai_surprise_bpb", comments="")
    print(f"\n{backend} backend: {len(wins)} windows, "
          f"mean AI-surprise = {np.nanmean(bpb):.3f} bits/base "
          f"(wrote ai_surprise{tag}.csv)")


if __name__ == "__main__":
    main()
