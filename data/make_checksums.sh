#!/usr/bin/env bash
# Regenerate checksums for whatever reference files are present in data/.
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob
files=(*.fa.gz *.fasta.gz *.fasta *.bed *.bb *.bw)
[ ${#files[@]} -eq 0 ] && { echo "no reference files in data/; run fetch.sh first"; exit 1; }
shasum -a 256 "${files[@]}"
