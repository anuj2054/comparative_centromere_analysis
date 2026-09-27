"""Fetch REAL sequence from NCBI for the analysis. Everything is cached to
data/ so downstream runs are reproducible and offline.

Sources (all real, all public):
  * T2T-CHM13 v2.0 chromosome 21 = RefSeq NC_060945.1
      - a multi-Mb region spanning the centromere for the descriptive map
      - centromeric fragments (satellite class) and q-arm fragments (unique class)
  * RefSeq curated human mRNAs -> CDS-only sequence (coding class)

Class labels are external to our metrics:
  - "coding"     : RefSeq CDS annotation
  - "satellite"  : centromeric region of chr21 (verified post-hoc by 171-bp periodicity)
  - "unique"     : single-copy q-arm region of chr21
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CHR21 = "NC_060945.1"  # T2T-CHM13v2.0 chromosome 21

# Region definitions on chr21 (1-based, inclusive). Verified empirically after fetch.
MAP_REGION = (9_000_000, 13_000_000)        # spans centromeric satellite + flanks
SATELLITE_REGION = (10_700_000, 11_300_000)  # within the chr21 centromere
UNIQUE_REGION = (33_000_000, 33_600_000)     # single-copy q-arm sequence

# Well-known human RefSeq mRNAs (CDS extracted) for the coding class.
CODING_MRNAS = [
    "NM_000518.5",   # HBB  - hemoglobin beta
    "NM_002046.7",   # GAPDH
    "NM_001101.5",   # ACTB - beta actin
    "NM_000546.6",   # TP53
    "NM_007294.4",   # BRCA1
    "NM_000207.3",   # INS  - insulin
    "NM_000314.8",   # PTEN
    "NM_005228.5",   # EGFR
    "NM_004985.5",   # KRAS
    "NM_000059.4",   # BRCA2
    "NM_000088.4",   # COL1A1
    "NM_001126112.3",# TP53 variant
]


def _get(params: dict, tries: int = 4) -> str:
    url = EFETCH + "?" + urllib.parse.urlencode(params)
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GenomeComplexity/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (t + 1))
    raise RuntimeError(f"efetch failed: {last}")


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    recs, name, seq = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(seq)))
            name, seq = line[1:].strip(), []
        elif line.strip():
            seq.append(line.strip())
    if name is not None:
        recs.append((name, "".join(seq)))
    return recs


def fetch_region(acc: str, start: int, stop: int, chunk: int = 500_000) -> str:
    """Fetch [start, stop] of an accession, chunked to be gentle on the API."""
    pieces = []
    s = start
    while s <= stop:
        e = min(s + chunk - 1, stop)
        txt = _get(dict(db="nuccore", id=acc, rettype="fasta", retmode="text",
                        seq_start=s, seq_stop=e))
        seq = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))
        pieces.append(seq)
        time.sleep(0.4)  # < 3 req/s
        s = e + 1
    return "".join(pieces)


def fetch_cds(mrna_ids: list[str]) -> list[tuple[str, str]]:
    """Fetch CDS-only nucleotide sequence for RefSeq mRNAs."""
    out = []
    for acc in mrna_ids:
        txt = _get(dict(db="nuccore", id=acc, rettype="fasta_cds_na", retmode="text"))
        for name, seq in _parse_fasta(txt):
            if seq:
                out.append((f"{acc}|{name.split()[0]}", seq))
        time.sleep(0.4)
    return out


def main():
    os.makedirs(DATA, exist_ok=True)
    manifest = {}

    print("Fetching CHM13 chr21 map region %s:%d-%d ..." % (CHR21, *MAP_REGION))
    map_seq = fetch_region(CHR21, *MAP_REGION)
    with open(os.path.join(DATA, "chr21_map.fasta"), "w") as f:
        f.write(f">{CHR21}:{MAP_REGION[0]}-{MAP_REGION[1]} CHM13v2.0 chr21 map region\n")
        f.write(map_seq + "\n")
    manifest["map"] = {"acc": CHR21, "region": MAP_REGION, "len": len(map_seq)}
    print("  got", len(map_seq), "bp")

    print("Fetching satellite region ...")
    sat = fetch_region(CHR21, *SATELLITE_REGION)
    with open(os.path.join(DATA, "satellite.fasta"), "w") as f:
        f.write(f">{CHR21}:{SATELLITE_REGION[0]}-{SATELLITE_REGION[1]} chr21 centromeric\n")
        f.write(sat + "\n")
    manifest["satellite"] = {"acc": CHR21, "region": SATELLITE_REGION, "len": len(sat)}
    print("  got", len(sat), "bp")

    print("Fetching unique q-arm region ...")
    uniq = fetch_region(CHR21, *UNIQUE_REGION)
    with open(os.path.join(DATA, "unique.fasta"), "w") as f:
        f.write(f">{CHR21}:{UNIQUE_REGION[0]}-{UNIQUE_REGION[1]} chr21 q-arm unique\n")
        f.write(uniq + "\n")
    manifest["unique"] = {"acc": CHR21, "region": UNIQUE_REGION, "len": len(uniq)}
    print("  got", len(uniq), "bp")

    print("Fetching coding CDS ...")
    cds = fetch_cds(CODING_MRNAS)
    with open(os.path.join(DATA, "coding_cds.fasta"), "w") as f:
        for name, seq in cds:
            f.write(f">{name}\n{seq}\n")
    manifest["coding"] = {"n_cds": len(cds), "total_bp": sum(len(s) for _, s in cds)}
    print("  got", len(cds), "CDS,", manifest["coding"]["total_bp"], "bp")

    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
