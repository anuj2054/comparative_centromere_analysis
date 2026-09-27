#!/usr/bin/env bash
# Process the HIGH-SALT CENP-A CUT&RUN pair as a sensitivity check.
#
# WHY. The functional label is built from the two LOW-SALT CENP-A replicates
# (SRR15395853 Expt 1, SRR15395851 Expt 2), chosen to match the published HG002 track,
# which is itself low-salt. BioProject PRJNA559484 also contains the high-salt pair:
#
#     SRR15395852  CENP-A CUT&RUN (high-salt) CHM13 Expt 1
#     SRR15395850  CENP-A CUT&RUN (high-salt) CHM13 Expt 2
#
# Whether the per-chromosome calls depend on the salt fraction was previously untested
# and recorded as an open item. This closes it.
#
# FULL DEPTH, deliberately. The low-salt track was built at full depth; subsampling the
# high-salt one would confound "does the fraction matter" with "does depth matter". The
# comparison is only clean if the processing is identical, so the recipe below is
# byte-for-byte the same as build_cenpa_track.sh: bwa mem -k 50 -c 1000000, q20 trim,
# samtools -F 3852, no MAPQ filter.
set -euo pipefail
TH="${1:-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="${CENPA_WORK:-$ROOT/work/cenpa}"; VENV="${CENPA_VENV:-$W/venv}"
REF="$W/chm13v2.0.fa"
OUT="$W/chm13v2.0.chm13_CA_cutnrun_hisalt_trimmed_q20.F3852.bw"
RUNS=(SRR15395852 SRR15395850)
cd "$W"

# Retry the METADATA call too. A previous run died because this timed out, returned
# empty, and the download loop below simply never executed -- so the guard inside it
# could not fire and execution fell through to trimming with no input.
#
# We ask for fastq_bytes alongside fastq_ftp so the download can tell a TRUNCATED file
# (resume it) from a CORRUPT one (start over). Emits one "url<TAB>bytes" line per mate.
ena () {
  local out=""
  for a in 1 2 3 4 5; do
    out=$(curl -fsSL --max-time 60 --retry 3 \
      "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$1&result=read_run&fields=fastq_ftp,fastq_bytes&format=tsv" \
      2>/dev/null \
      | awk -F'\t' 'NR>1{n=split($(NF-1),u,";"); split($NF,b,";");
                        for(i=1;i<=n;i++) if(u[i]~/fastq\.gz$/) print "https://"u[i]"\t"b[i]}' || true)
    [ -n "$out" ] && { echo "$out"; return 0; }
    sleep $((a*10))
  done
  return 1
}

# Fetch one file to its exact ENA byte count, RESUMING a short partial instead of
# discarding it. The previous version deleted anything failing gzip -t, which threw away
# gigabytes of perfectly good prefix every time a transfer closed early -- and ENA closes
# transfers early often enough that a from-scratch retry may never converge.
grab () {
  local u="$1" want="$2" f="$3" have
  for try in 1 2 3 4 5 6; do
    have=0; [ -e "$f" ] && have=$(wc -c < "$f" | tr -d ' ')
    if [ "$have" -eq "$want" ]; then
      gzip -t "$f" 2>/dev/null && { echo "  verified $(basename "$f") ($have bytes)"; return 0; }
      echo "  $(basename "$f") is the expected size but fails gzip -t, restarting it"
      rm -f "$f"; have=0
    elif [ "$have" -gt "$want" ]; then
      echo "  $(basename "$f") is larger than ENA reports ($have > $want), restarting it"
      rm -f "$f"; have=0
    fi
    echo "  fetching $(basename "$f") attempt $try, have $have of $want bytes"
    curl -L --no-progress-meter -C - --retry 8 --retry-delay 15 --retry-all-errors \
         --speed-time 120 --speed-limit 10000 -o "$f" "$u" || true
  done
  echo "DOWNLOAD FAILED: $f"; return 1
}

for acc in "${RUNS[@]}"; do
  echo "== $acc =="
  [ -s "$acc.filt.bam" ] && { echo "  already aligned"; continue; }
  meta=$(ena "$acc") || { echo "ENA METADATA FAILED for $acc after 5 attempts"; exit 1; }
  n_urls=$(echo "$meta" | wc -l | tr -d ' ')
  [ "$n_urls" -eq 2 ] || { echo "expected 2 FASTQ urls for $acc, got $n_urls"; exit 1; }
  i=1
  while IFS=$'\t' read -r u want; do
    grab "$u" "$want" "${acc}_${i}.fastq.gz" || exit 1
    i=$((i+1))
  done <<< "$meta"
  "$VENV/bin/cutadapt" -j "$TH" -q 20 -m 25 -a AGATCGGAAGAGC -A AGATCGGAAGAGC \
      -o "${acc}_1.hstrim.fq.gz" -p "${acc}_2.hstrim.fq.gz" \
      "${acc}_1.fastq.gz" "${acc}_2.fastq.gz" > "$acc.cutadapt.log"
  echo "  $acc: bwa mem (identical recipe to the low-salt track)"
  bwa mem -t "$TH" -k 50 -c 1000000 "$REF" "${acc}_1.hstrim.fq.gz" "${acc}_2.hstrim.fq.gz" \
      2> "$acc.bwa.log" \
    | samtools view -@ 2 -b -F 3852 - \
    | samtools sort -@ 3 -m 1G -T "$acc.st" -o "$acc.filt.bam" -
  samtools index "$acc.filt.bam"
  rm -f "${acc}_1.fastq.gz" "${acc}_2.fastq.gz"
done

echo "== merge + bigWig =="
M="${OUT%.bw}.bam"
[ -s "$M" ] || { samtools merge -@ "$TH" -f "$M" "${RUNS[0]}.filt.bam" "${RUNS[1]}.filt.bam"; samtools index "$M"; }
[ -s "$OUT" ] || "$VENV/bin/bamCoverage" -b "$M" -o "$OUT" -p "$TH" --binSize 1000 \
     --normalizeUsing CPM --extendReads
echo "== merged low+high (the third condition) =="
# The UCSC hub titles the published track "low salt only" but its Methods paragraph says
# "High and low salt data was merged following mapping, prior to kmer filtering". Those
# are inconsistent, and the merged condition is the one that decides which reading the
# published track actually corresponds to. It needs no new alignment: the two low-salt
# BAMs already exist from build_cenpa_track.sh, and the two high-salt ones from here.
MERGED="$W/chm13v2.0.chm13_CA_cutnrun_merged_trimmed_q20.F3852.bam"
MOUT="${MERGED%.bam}.bw"
if [ ! -s "$MERGED" ]; then
  for b in SRR15395853 SRR15395851 SRR15395852 SRR15395850; do
    [ -s "$b.filt.bam" ] || { echo "missing $b.filt.bam, cannot build merged"; exit 1; }
  done
  samtools merge -@ "$TH" -f "$MERGED" \
    SRR15395853.filt.bam SRR15395851.filt.bam SRR15395852.filt.bam SRR15395850.filt.bam
  samtools index "$MERGED"
fi
[ -s "$MOUT" ] || "$VENV/bin/bamCoverage" -b "$MERGED" -o "$MOUT" -p "$TH" --binSize 1000 \
     --normalizeUsing CPM --extendReads
echo "  wrote $MOUT"

echo "DONE $(date -u +%FT%TZ)"
echo "  low-salt : $W/chm13v2.0.chm13_CA_cutnrun_losalt_trimmed_q20.F3852.bw"
echo "  high-salt: $OUT"
echo "  merged   : $MOUT"
