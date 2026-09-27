"""Remote Evo 2 inference runner (executed on the GPU instance).

Computes per-base negative log-likelihood ("AI-surprise", bits/base) from Evo 2
for (a) the labeled class fragments and (b) sliding windows across the chr21 map
region, with fragmentation IDENTICAL to src/build_chr21_map.py so results merge by
(label, frag) and by window start coordinate.

Outputs:
  out/evo2_class.csv   label,frag,evo2_bpb
  out/evo2_map.csv     start,end,evo2_bpb
"""
import csv
import json
import math
import os
import sys
import time

import torch

DATA = os.path.expanduser("~/gc/data")
OUT = os.path.expanduser("~/gc/out")
WINDOW = 2000
MAP_STEP = 4000
MAX_FRAGS = 120
MAP_MAX = 350            # subsample map windows to bound runtime
MAX_LEN = 8192          # cap context (windows are 2 kb; ample headroom)


def clean(seq):
    return "".join(c for c in seq.upper() if c in "ACGT")


def load_fasta_concat(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


def load_cds_concat(path):
    out = []
    for line in open(path):
        if not line.startswith(">"):
            out.append(clean(line.strip()))
    return "".join(out)


def frags(seq, w=WINDOW, n=MAX_FRAGS):
    s = clean(seq)
    return [s[i * w:(i + 1) * w] for i in range(min(n, len(s) // w))]


def load_model():
    from evo2 import Evo2
    name = os.environ.get("EVO2_MODEL", "evo2_7b")
    print(f"loading {name} ...", flush=True)
    t = time.time()
    model = Evo2(name)
    print(f"loaded in {time.time()-t:.0f}s", flush=True)
    return model, name


@torch.no_grad()
def surprise_bpb(model, seq):
    """Mean next-token NLL in bits/base over the sequence."""
    seq = seq[:MAX_LEN]
    ids = model.tokenizer.tokenize(seq)
    t = torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
    out = model(t)
    # Evo2 returns nested ((logits, None), None); unwrap to the first tensor.
    node = out
    while isinstance(node, (tuple, list)):
        node = node[0]
    logits = node[0]                        # (1, L, V) -> (L, V)
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = t[0, 1:]
    nll = -logp[torch.arange(tgt.shape[0], device=logp.device), tgt]
    return float(nll.mean().item() / math.log(2))


def main():
    os.makedirs(OUT, exist_ok=True)
    model, name = load_model()

    # --- class fragments ---
    classes = {
        "satellite": frags(load_fasta_concat(f"{DATA}/satellite.fasta")),
        "unique": frags(load_fasta_concat(f"{DATA}/unique.fasta")),
        "coding": frags(load_cds_concat(f"{DATA}/coding_cds.fasta")),
    }
    with open(f"{OUT}/evo2_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "frag", "evo2_bpb", "model"])
        for label, seqs in classes.items():
            t = time.time()
            for i, s in enumerate(seqs):
                w.writerow([label, i, f"{surprise_bpb(model, s):.6f}", name])
            f.flush()
            print(f"{label}: {len(seqs)} frags in {time.time()-t:.0f}s", flush=True)

    # --- map windows (subsampled, identical coords to build_chr21_map) ---
    man = json.load(open(f"{DATA}/manifest.json"))
    offset = man["map"]["region"][0]
    mseq = clean(load_fasta_concat(f"{DATA}/chr21_map.fasta"))
    starts = list(range(0, len(mseq) - WINDOW + 1, MAP_STEP))
    stride = max(1, len(starts) // MAP_MAX)
    starts = starts[::stride]
    with open(f"{OUT}/evo2_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "evo2_bpb"])
        t = time.time()
        for j, s in enumerate(starts):
            seg = mseq[s:s + WINDOW]
            w.writerow([offset + s, offset + s + WINDOW, f"{surprise_bpb(model, seg):.6f}"])
            if j % 50 == 0:
                f.flush()
                print(f"  map {j}/{len(starts)}", flush=True)
        print(f"map: {len(starts)} windows in {time.time()-t:.0f}s", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
