#!/usr/bin/env bash
# Orthogonal functional validation with CHM13 CENP-A ChIP-seq + matched input.
#
# WHY. The label built by build_cenpa_track.sh comes from CENP-A CUT&RUN (SRR15395851
# + SRR15395853, low-salt,
# both replicates -- verified against SRA titles and the BAM @PG header). That is the
# right choice for matching the published HG002 losalt track, but it is one assay with
# no input normalisation. PRJNA559484 also contains a completely orthogonal, well
# documented CHM13 CENP-A ChIP-seq with its own matched input:
#
#     SRR13278683, SRR13278684   CENP-A ChIP-Seq of CHM13hTERT
#     SRR13278681, SRR13278682   Input of CHM13hTERT   (matched control)
#
# If the ChIP-seq reproduces the CUT&RUN calls, the functional label is assay-independent
# and the 19/19 result is far better supported than by either assay alone. If it does not,
# we need to know before submitting.
#
# Input is subsampled 1-in-4: it is a background estimate consumed as an array-level mean
# over megabase intervals, where depth affects variance and not bias. CENP-A is full depth.
# Alignment recipe is identical to build_cenpa_track.sh so the two are comparable.
set -euo pipefail
TH="${1:-8}"; FRAC="${FRAC:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="${CENPA_WORK:-$ROOT/work/cenpa}"; VENV="${CENPA_VENV:-$W/venv}"
REF="$W/chm13v2.0.fa"
CHIP=(SRR13278683 SRR13278684)      # CENP-A, full depth
INPUT=(SRR13278681 SRR13278682)     # matched input, subsampled
cd "$W"

# ENA metadata, WITH RETRY. Without one a single timeout returns empty, the download
# loop below iterates zero times, fetch() returns success having fetched nothing, and
# align() walks straight into bwa with no input. That is how a previous run died.
#
# We ask for fastq_bytes alongside fastq_ftp so grab() can tell a TRUNCATED file
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
# the whole prefix every time a transfer closed early -- and ENA closes transfers early
# often enough that a from-scratch retry may never converge on a 3 GB file.
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

fetch () {                              # $1 = accession
  local acc="$1" i=1 meta n_urls
  meta=$(ena "$acc") || { echo "ENA METADATA FAILED for $acc after 5 attempts"; exit 1; }
  n_urls=$(echo "$meta" | wc -l | tr -d ' ')
  [ "$n_urls" -eq 2 ] || { echo "expected 2 FASTQ urls for $acc, got $n_urls"; exit 1; }
  while IFS=$'\t' read -r u want; do
    grab "$u" "$want" "${acc}_${i}.fastq.gz" || exit 1
    i=$((i+1))
  done <<< "$meta"
}

align () {                              # $1 = accession, $2 = subsample? (yes/no)
  local acc="$1" sub="$2"
  [ -s "$acc.filt.bam" ] && { echo "  $acc already aligned"; return; }
  fetch "$acc"
  local r1="${acc}_1.fastq.gz" r2="${acc}_2.fastq.gz"
  if [ "$sub" = yes ]; then
    for m in 1 2; do
      local s="${acc}_${m}.sub.fq.gz"
      [ -s "$s" ] && gzip -t "$s" 2>/dev/null && continue
      echo "  subsampling $acc mate $m (1 in $FRAC)"
      gunzip -c "${acc}_${m}.fastq.gz" | awk -v n="$FRAC" 'NR%(4*n)>=1 && NR%(4*n)<=4' | gzip > "$s"
    done
    n1=$(gunzip -c "${acc}_1.sub.fq.gz" | wc -l); n2=$(gunzip -c "${acc}_2.sub.fq.gz" | wc -l)
    [ "$n1" = "$n2" ] || { echo "MATE MISMATCH $acc: $n1 vs $n2"; exit 1; }
    r1="${acc}_1.sub.fq.gz"; r2="${acc}_2.sub.fq.gz"
  fi
  "$VENV/bin/cutadapt" -j "$TH" -q 20 -m 25 -a AGATCGGAAGAGC -A AGATCGGAAGAGC \
      -o "${acc}_1.ctrim.fq.gz" -p "${acc}_2.ctrim.fq.gz" "$r1" "$r2" > "$acc.cutadapt.log"
  echo "  $acc: bwa mem (identical recipe to build_cenpa_track.sh)"
  bwa mem -t "$TH" -k 50 -c 1000000 "$REF" "${acc}_1.ctrim.fq.gz" "${acc}_2.ctrim.fq.gz" \
      2> "$acc.bwa.log" \
    | samtools view -@ 2 -b -F 3852 - \
    | samtools sort -@ 3 -m 1G -T "$acc.st" -o "$acc.filt.bam" -
  samtools index "$acc.filt.bam"
}

echo "== CENP-A ChIP-seq (full depth) =="
for a in "${CHIP[@]}";  do align "$a" no;  done
echo "== matched input (1 in $FRAC) =="
for a in "${INPUT[@]}"; do align "$a" yes; done

for tag in chip input; do
  if [ "$tag" = chip ]; then set -- "${CHIP[@]}"; else set -- "${INPUT[@]}"; fi
  M="$W/chm13v2.0.chm13_CA_${tag}_q20.F3852.bam"; O="${M%.bam}.bw"
  [ -s "$M" ] || { samtools merge -@ "$TH" -f "$M" "$1.filt.bam" "$2.filt.bam"; samtools index "$M"; }
  [ -s "$O" ] || "$VENV/bin/bamCoverage" -b "$M" -o "$O" -p "$TH" --binSize 1000 \
       --normalizeUsing CPM --extendReads
  echo "  wrote $O"
done
echo "DONE $(date -u +%FT%TZ)"
