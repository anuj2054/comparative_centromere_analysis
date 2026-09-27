"""Remote Evo 2 runner for the 5 new chromosomes' centromere windows.

Reads ~/gc/multichrom_windows.csv (chrom,start,end,...), re-fetches each
chromosome region from NCBI, slices the exact windows by coordinate, and
computes Evo 2 (`evo2_7b`, bf16) next-token surprise (bits/base) per window.
Output: ~/gc/out/evo2_multichrom.csv  (chrom,start,evo2_bpb)
"""
import csv
import math
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import torch

DATA = os.path.expanduser("~/gc/multichrom_windows.csv")
OUT = os.path.expanduser("~/gc/out/evo2_multichrom.csv")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# CHM13 RefSeq accessions: chrN = NC_0609{24+N}.1; chrX is the 23rd record
# (NC_060947.1). NC_060948.1 is chrY (HG002/NA24385), a different individual.
ACC = {"chr13": "NC_060937.1", "chr14": "NC_060938.1", "chr15": "NC_060939.1",
       "chr22": "NC_060946.1", "chrX": "NC_060947.1"}
MAX_LEN = 8192


def fetch_region(acc, start, stop, chunk=500_000):
    out = []
    s = start
    while s <= stop:
        e = min(s + chunk - 1, stop)
        url = EFETCH + "?" + urllib.parse.urlencode(dict(
            db="nuccore", id=acc, rettype="fasta", retmode="text",
            seq_start=s, seq_stop=e))
        req = urllib.request.Request(url, headers={"User-Agent": "GC/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = r.read().decode()
        out.append("".join(l.strip() for l in txt.splitlines() if not l.startswith(">")))
        time.sleep(0.34)
        s = e + 1
    return "".join(out)


def load_model():
    from evo2 import Evo2
    print("loading evo2_7b ...", flush=True)
    t = time.time()
    m = Evo2("evo2_7b")
    print(f"loaded in {time.time()-t:.0f}s", flush=True)
    return m


@torch.no_grad()
def surprise_bpb(model, seq):
    seq = seq[:MAX_LEN]
    ids = model.tokenizer.tokenize(seq)
    t = torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
    out = model(t)
    node = out
    while isinstance(node, (tuple, list)):
        node = node[0]
    logits = node[0]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = t[0, 1:]
    nll = -logp[torch.arange(tgt.shape[0], device=logp.device), tgt]
    return float(nll.mean().item() / math.log(2))


def main():
    rows = list(csv.DictReader(open(DATA)))
    by_chrom = defaultdict(list)
    for r in rows:
        by_chrom[r["chrom"]].append((int(r["start"]), int(r["end"])))
    model = load_model()

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chrom", "start", "evo2_bpb"])
        for chrom, wins in by_chrom.items():
            acc = ACC[chrom]
            lo = min(s for s, e in wins)
            hi = max(e for s, e in wins)
            print(f"[{chrom}] fetching {acc}:{lo}-{hi} ({len(wins)} windows) ...", flush=True)
            region = fetch_region(acc, lo, hi)
            t = time.time()
            for i, (s, e) in enumerate(wins):
                seg = region[s - lo:e - lo]
                seg = "".join(c for c in seg.upper() if c in "ACGT")
                val = surprise_bpb(model, seg) if len(seg) >= 50 else float("nan")
                w.writerow([chrom, s, f"{val:.6f}"])
                if i % 100 == 0:
                    f.flush()
                    print(f"  [{chrom}] {i}/{len(wins)}", flush=True)
            print(f"[{chrom}] done in {time.time()-t:.0f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
