#!/usr/bin/env bash
# Build the matched IgG control track and re-derive the functional calls as
# CENP-A/IgG ratios.
#
# WHY. The CENP-A track from build_cenpa_track.sh is raw coverage with no input
# control, so array-level
# signal could in principle reflect mappability or accessibility rather than CENP-A
# binding. The matched controls sit in the same BioProject and the same cell line and
# were simply overlooked:
#     SRR15395849  IgG CUT&RUN (low-salt) CHM13 Expt 1   pairs with SRR15395853
#     SRR15395847  IgG CUT&RUN (low-salt) CHM13 Expt 2   pairs with SRR15395851
#
# SUBSAMPLING. IgG is a diffuse background estimate and every downstream use is an
# array-level mean over megabase intervals, where depth affects variance not bias.
# We take every 4th read pair -- deterministic, and immune to any ordering structure
# in the file, unlike taking a prefix. 25% still gives ~3x genome coverage.
#
# Everything else mirrors the CENP-A recipe exactly: bwa mem -k 50 -c 1000000,
# q20 pre-alignment trimming, samtools -F 3852, no MAPQ filter.
set -euo pipefail
TH="${1:-8}"; FRAC="${FRAC:-4}"          # keep 1 pair in FRAC
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="${CENPA_WORK:-$ROOT/work/cenpa}"; VENV="${CENPA_VENV:-$W/venv}"
REF="$W/chm13v2.0.fa"; OUT="$W/chm13v2.0.chm13_IgG_cutnrun_losalt_q20.F3852.bw"
RUNS=(SRR15395849 SRR15395847)
cd "$W"

ena () { curl -fsSL "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$1&result=read_run&fields=fastq_ftp&format=tsv" \
         | awk -F'\t' 'NR>1{print $NF}' | tr ';' '\n' | grep fastq.gz | sed 's|^|https://|'; }

for acc in "${RUNS[@]}"; do
  echo "== $acc =="
  [ -s "$acc.filt.bam" ] && { echo "  already aligned"; continue; }
  i=1
  for u in $(ena "$acc"); do
    f="${acc}_${i}.fastq.gz"
    for try in 1 2 3 4 5; do
      [ -s "$f" ] && gzip -t "$f" 2>/dev/null && break
      echo "  downloading $(basename "$u") (attempt $try)"
      # keep the partial file and resume with -C -; only discard it if it is corrupt
      [ -s "$f" ] && ! gzip -t "$f" 2>/dev/null && rm -f "$f"
      curl -L --no-progress-meter -C - --retry 8 --retry-delay 15 --retry-all-errors \
           --speed-time 120 --speed-limit 10000 -o "$f" "$u" || true
    done
    if [ ! -s "$f" ]; then echo "DOWNLOAD FAILED after 3 attempts: $f"; exit 1; fi
    gzip -t "$f" 2>/dev/null || { echo "CORRUPT after 3 attempts: $f"; exit 1; }
    echo "  verified $f ($(du -h "$f" | cut -f1))"
    i=$((i+1))
  done
  # keep every FRAC-th pair, in lockstep across mates so pairing is preserved
  for m in 1 2; do
    s="${acc}_${m}.sub.fq.gz"
    [ -s "$s" ] && gzip -t "$s" 2>/dev/null && continue
    [ -s "${acc}_${m}.fastq.gz" ] || { echo "MISSING INPUT: ${acc}_${m}.fastq.gz"; exit 1; }
    echo "  subsampling mate $m (1 in $FRAC)"
    gunzip -c "${acc}_${m}.fastq.gz" \
      | awk -v n="$FRAC" 'NR%(4*n)>=1 && NR%(4*n)<=4' | gzip > "$s"
  done
  n1=$(gunzip -c "${acc}_1.sub.fq.gz" | wc -l); n2=$(gunzip -c "${acc}_2.sub.fq.gz" | wc -l)
  [ "$n1" = "$n2" ] || { echo "MATE MISMATCH after subsampling: $n1 vs $n2"; exit 1; }
  echo "  $((n1/4)) pairs kept"
  "$VENV/bin/cutadapt" -j "$TH" -q 20 -m 25 -a AGATCGGAAGAGC -A AGATCGGAAGAGC \
      -o "${acc}_1.subtrim.fq.gz" -p "${acc}_2.subtrim.fq.gz" \
      "${acc}_1.sub.fq.gz" "${acc}_2.sub.fq.gz" > "$acc.cutadapt.log"
  echo "  aligning (same recipe as CENP-A)"
  bwa mem -t "$TH" -k 50 -c 1000000 "$REF" "${acc}_1.subtrim.fq.gz" "${acc}_2.subtrim.fq.gz" \
      2> "$acc.bwa.log" \
    | samtools view -@ 2 -b -F 3852 - \
    | samtools sort -@ 3 -m 1G -T "$acc.st" -o "$acc.filt.bam" -
  samtools index "$acc.filt.bam"
done

echo "== merge + bigWig =="
M="${OUT%.bw}.bam"
[ -s "$M" ] || { samtools merge -@ "$TH" -f "$M" "${RUNS[0]}.filt.bam" "${RUNS[1]}.filt.bam"; samtools index "$M"; }
"$VENV/bin/bamCoverage" -b "$M" -o "$OUT" -p "$TH" --binSize 1000 --normalizeUsing CPM --extendReads
echo "DONE $OUT"
