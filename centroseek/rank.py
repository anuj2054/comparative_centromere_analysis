"""Rank candidate alpha-satellite HOR arrays within each chromosome.

    python -m centroseek.rank --fasta genome.fa --arrays candidates.bed

The BED gives the candidate arrays: chrom, start, end and optionally a name in
column 4. Every candidate on a chromosome competes only against the others on that
same chromosome, which is the task the model was built for.

Two rules are reported side by side, deliberately. Across the manuscript's data the
parameter-free longest-array rule is the more accurate of the two (19/19 human and
44/47 ape, against 17/19 and 38/47 for CentroSeek). CentroSeek's value is that it
recovers the three bonobo chromosomes on which length fails. Disagreement between
the columns is the interesting signal, not a defect.
"""
import argparse
import json
import os
import sys
from math import exp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from centroseek.features import array_features  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")


def load_model(path=MODEL_PATH):
    with open(path) as fh:
        return json.load(fh)


def score(model, bz2_bpb, h11):
    """Logistic probability from the standardised features."""
    z = model["intercept"]
    for v, m, s, c in zip((bz2_bpb, h11), model["standardisation"]["mean"],
                          model["standardisation"]["scale"], model["coefficients"]):
        z += ((v - m) / s) * c
    return 1.0 / (1.0 + exp(-z))


def read_bed(path):
    """chrom -> list of (start, end, name). BED is half-open and 0-based."""
    by_chrom = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            chrom, start, end = f[0], int(f[1]), int(f[2])
            name = f[3] if len(f) > 3 else f"{chrom}:{start}-{end}"
            by_chrom.setdefault(chrom, []).append((start, end, name))
    return by_chrom


def extract(fasta, wanted):
    """Pull just the requested intervals, holding one record in memory at a time."""
    out = {}
    cur, buf = None, []

    def flush():
        if cur in wanted:
            seq = "".join(buf)
            for s, e, name in wanted[cur]:
                out[(cur, name)] = seq[s:e]

    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                cur, buf = line[1:].split()[0], []
            elif cur in wanted:
                buf.append(line.strip())
    flush()
    return out


def rank_arrays(fasta, bed, model=None, min_span=10000, min_windows=5):
    """Return one row per candidate array, ranked within its chromosome."""
    model = model or load_model()
    wanted = read_bed(bed)
    seqs = extract(fasta, wanted)

    rows = []
    for chrom, cands in wanted.items():
        for start, end, name in cands:
            seq = seqs.get((chrom, name), "")
            if not seq:
                print(f"warning: {chrom} {name} not found in FASTA, skipped",
                      file=sys.stderr)
                continue
            span = end - start
            bz2_bpb, h11, n_win = array_features(seq)
            passes = span >= min_span and n_win >= min_windows
            rows.append(dict(
                chrom=chrom, array=name, start=start, end=end, span_bp=span,
                n_windows=n_win, bz2_bpb=bz2_bpb, H11=h11,
                centroseek_p=score(model, bz2_bpb, h11) if bz2_bpb is not None else None,
                passes_threshold=passes,
            ))

    # Rank within chromosome, over the arrays that clear the threshold.
    for chrom in {r["chrom"] for r in rows}:
        grp = [r for r in rows if r["chrom"] == chrom and r["passes_threshold"]]
        for key, field in (("centroseek_p", "centroseek_rank"), ("span_bp", "length_rank")):
            for i, r in enumerate(sorted(grp, key=lambda r: -r[key]), 1):
                r[field] = i
        for r in rows:
            if r["chrom"] == chrom and not r["passes_threshold"]:
                r["centroseek_rank"] = r["length_rank"] = None
    return rows


COLUMNS = ["chrom", "array", "start", "end", "span_bp", "n_windows", "bz2_bpb",
           "H11", "centroseek_p", "centroseek_rank", "length_rank", "passes_threshold"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="genome FASTA")
    ap.add_argument("--arrays", required=True, help="BED of candidate arrays")
    ap.add_argument("--out", help="output TSV (default: stdout)")
    ap.add_argument("--min-span", type=int, default=10000,
                    help="minimum array span in bp (default 10000, as in the paper)")
    ap.add_argument("--min-windows", type=int, default=5,
                    help="minimum scored windows (default 5, as in the paper)")
    a = ap.parse_args()

    rows = rank_arrays(a.fasta, a.arrays, min_span=a.min_span, min_windows=a.min_windows)
    fh = open(a.out, "w") if a.out else sys.stdout
    print("\t".join(COLUMNS), file=fh)
    for r in sorted(rows, key=lambda r: (r["chrom"], r["start"])):
        print("\t".join("" if r.get(c) is None else
                        (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r[c]))
                        for c in COLUMNS), file=fh)
    if a.out:
        fh.close()

    calls = {}
    for r in rows:
        if r.get("centroseek_rank") == 1:
            calls.setdefault(r["chrom"], {})["centroseek"] = r["array"]
        if r.get("length_rank") == 1:
            calls.setdefault(r["chrom"], {})["longest"] = r["array"]
    dis = [c for c, v in calls.items() if v.get("centroseek") != v.get("longest")]
    print(f"\n{len(calls)} chromosome(s) ranked; the two rules disagree on "
          f"{len(dis)}{': ' + ', '.join(sorted(dis)) if dis else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
