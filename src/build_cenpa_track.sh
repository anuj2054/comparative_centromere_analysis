#!/usr/bin/env bash
# Build a CHM13-native CENP-A track from raw reads, and re-run the functional analysis.
# Answers R1#4 and R2#6.
#
# The published functional label uses HG002 CENP-A CUT&RUN mapped onto CHM13, a
# cross-individual proxy. Both reviewers object, and R1 notes chr4 has a different
# active array in CHM13 than in HG002. CHM13's own CENP-A exists only as raw reads
# (BioProject PRJNA559484) -- this builds the track from them.
#
# The existing HG002 track is
#   chm13v2.0.hg002_CA_cutnrun_losalt_trimmed_q20_2.F3852.bw
# and the published recipe behind it (UCSC T2T hub, cutnrun-losalt track description;
# mapping and samtools filtering by K. Miga) is verbatim:
#
#   "Reads were trimmed and filtered (q20) before mapped with bwa mem -k 50 -c 1000000
#    and samtools filtered (F3852) for properly paired reads"
#
# So: LOW-SALT fraction, trimmed, q20, bwa mem -k 50 -c 1000000, samtools -F 3852.
# Every step below mirrors that, because the comparison to published numbers is only
# meaningful if the processing matches. Do not "improve" the filters here.
#
# The hub also documents an optional extra stage -- Meryl genome-wide unique k-mer
# filtering (21/51/100-mers) -- which produced the sibling *_51mer.bw track. The track
# this project uses is the unfiltered F3852 one, so that stage is deliberately omitted.
#
# Tuned for an Apple-silicon MacBook: 16 GB RAM, Homebrew toolchain, FASTQs pulled
# straight from ENA (no SRA toolkit, no uncompressed intermediates).
#
#   IMPORTANT: use classic `bwa`, not `bwa-mem2`. bwa-mem2's index needs ~28 GB and
#   will not fit in 16 GB. bwa's needs ~4.5 GB.
#
# Usage:
#   bash src/build_cenpa_track.sh setup     # install toolchain (once)
#   bash src/build_cenpa_track.sh check     # preflight: tools, disk, RAM
#   bash src/build_cenpa_track.sh run [N]   # do it (N threads, default 8)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${CENPA_WORK:-$ROOT/work/cenpa}"
REF="$WORK/chm13v2.0.fa"
OUT="$WORK/chm13v2.0.chm13_CA_cutnrun_losalt_trimmed_q20.F3852.bw"
VENV="${CENPA_VENV:-$WORK/venv}"

RUNS=(SRR15395851 SRR15395853)          # low-salt CENP-A CUT&RUN, Expt 2 and Expt 1
CHIP=(SRR13278683 SRR13278684)          # CENP-A ChIP-seq, same cell line (CENPA_CHIP=1)
NEED_GB=150

say () { printf '\n== %s ==\n' "$*"; }

# ---------------------------------------------------------------- setup ----
cmd_setup () {
  say "toolchain"
  if ! command -v brew >/dev/null; then
    echo "Homebrew not found. Install it first (needs Xcode command line tools):"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
  fi
  brew install bwa samtools fastp
  mkdir -p "$WORK"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  # deeptools gives bamCoverage; if it fails to build, the pure-python fallback
  # build_bigwig_from_bam.py needs only pysam + pyBigWig and is used automatically.
  "$VENV/bin/pip" install -q pysam pyBigWig numpy || true
  "$VENV/bin/pip" install -q deeptools || echo "  deeptools unavailable -- will use the fallback"
  echo "toolchain ready"
}

# ------------------------------------------------------------- preflight ----
cmd_check () {
  local ok=0
  say "tools"
  for t in curl bwa samtools; do
    if command -v "$t" >/dev/null; then printf '  %-10s %s\n' "$t" "$(command -v $t)"
    else printf '  %-10s MISSING\n' "$t"; ok=1; fi
  done
  if command -v fastp >/dev/null; then printf '  %-10s %s\n' fastp "$(command -v fastp)"
  elif [[ -x "$VENV/bin/cutadapt" ]]; then printf '  %-10s %s (fastp substitute)\n' cutadapt "$VENV/bin/cutadapt"
  else echo "  trimmer   MISSING -- run 'setup' or src/build_alignment_tools.sh"; ok=1; fi
  if [[ -x "$VENV/bin/bamCoverage" ]]; then echo "  bamCoverage $VENV/bin/bamCoverage"
  elif [[ -x "$VENV/bin/python" ]]; then echo "  bamCoverage MISSING -- fallback build_bigwig_from_bam.py will be used"
  else echo "  bamCoverage MISSING and no venv -- run 'setup'"; ok=1; fi

  say "resources"
  local avail; avail=$(df -g "$ROOT" | awk 'NR==2{print $4}')
  echo "  free disk : ${avail} GB (need ~${NEED_GB} GB)"
  [[ "$avail" -lt "$NEED_GB" ]] && { echo "  INSUFFICIENT DISK"; ok=1; }
  echo "  RAM       : $(( $(sysctl -n hw.memsize) / 1073741824 )) GB (bwa needs ~5 GB; do NOT use bwa-mem2)"
  echo "  cores     : $(sysctl -n hw.ncpu) (use 6-8 threads; more will thermally throttle)"

  say "ENA availability"
  for r in "${RUNS[@]}"; do
    local n; n=$(curl -sS --max-time 30 \
      "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=fastq_ftp&format=tsv" \
      | tail -n +2 | tr ';' '\n' | grep -c fastq.gz || true)
    printf '  %-12s %s FASTQ files\n' "$r" "$n"
    [[ "$n" -lt 2 ]] && { echo "    NOT AVAILABLE"; ok=1; }
  done
  [[ "$ok" -eq 0 ]] && echo -e "\npreflight OK" || echo -e "\npreflight FAILED"
  return "$ok"
}

# ------------------------------------------------------------------ run ----
ena_urls () {                        # $1 = accession -> two https URLs
  # ENA always prepends a run_accession column to the requested fields, so the TSV
  # row is "SRR15395851<TAB>path_1.fastq.gz;path_2.fastq.gz". Without `cut -f2` the
  # accession is glued onto the first path and the mate-1 URL comes out as
  # "https://SRR15395851<TAB>ftp.sra.ebi.ac.uk/..." -- which curl cannot fetch, and
  # which word-splits into a bogus third "URL". Take the fastq_ftp column only.
  curl -sS --max-time 60 \
    "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$1&result=read_run&fields=fastq_ftp&format=tsv" \
    | awk -F'\t' 'NR>1{print $NF}' | tr ';' '\n' | grep fastq.gz | sed 's|^|https://|'
}

fetch_reads () {                     # $1 = accession
  local acc="$1" i=1
  for u in $(ena_urls "$acc"); do
    local f="$WORK/${acc}_${i}.fastq.gz"
    # Verify gzip integrity and retry. The AWS run died on
    #   gzip: SRR15395851_2.fq.gz: invalid compressed data--crc error
    # because a damaged transfer was only detected three stages later.
    for try in 1 2 3; do
      [[ -s "$f" ]] && gzip -t "$f" 2>/dev/null && break
      echo "  downloading $(basename "$u") (attempt $try)"
      [[ $try -gt 1 ]] && rm -f "$f"          # a resume onto a corrupt file cannot heal it
      curl -L --no-progress-meter -C - --retry 5 --retry-delay 10 -o "$f" "$u" || true
    done
    gzip -t "$f" 2>/dev/null || { echo "CORRUPT DOWNLOAD after 3 attempts: $f"; exit 1; }
    echo "  verified $(basename "$f") ($(du -h "$f" | cut -f1))"
    i=$((i+1))
  done
}

align_one () {                       # $1 = accession, $2 = threads
  local acc="$1" th="$2"
  [[ -s "$WORK/$acc.filt.bam" ]] && { echo "  $acc already aligned"; return; }
  fetch_reads "$acc"
  if [[ ! -s "$WORK/${acc}_1.trim.fq.gz" ]]; then
    echo "  $acc: trimming (adapter + q20), the published pre-alignment filter"
    if command -v fastp >/dev/null; then
      fastp -w "$(( th > 16 ? 16 : th ))" -q 20 -u 40 \
            -i "$WORK/${acc}_1.fastq.gz" -I "$WORK/${acc}_2.fastq.gz" \
            -o "$WORK/${acc}_1.trim.fq.gz" -O "$WORK/${acc}_2.trim.fq.gz" \
            -j "$WORK/$acc.fastp.json" -h "$WORK/$acc.fastp.html"
    else
      # cutadapt substitute: fastp needs libdeflate/isa-l and will not build without
      # admin here. Neither tool is named in the published methods, so this is free.
      # -q 20 quality-trim, -m 25 drop runt reads, universal Illumina adapter.
      "$VENV/bin/cutadapt" -j "$th" -q 20 -m 25 \
            -a AGATCGGAAGAGC -A AGATCGGAAGAGC \
            -o "$WORK/${acc}_1.trim.fq.gz" -p "$WORK/${acc}_2.trim.fq.gz" \
            "$WORK/${acc}_1.fastq.gz" "$WORK/${acc}_2.fastq.gz" > "$WORK/$acc.cutadapt.log"
    fi
  fi
  echo "  $acc: bwa mem -> MAPQ>=20 -F 3852 -> sort   (hours; safe to leave)"
  # -k 50 -c 1000000 are the PUBLISHED parameters, not bwa defaults (k=19, c=500).
  # -k 50 demands a 50 bp exact seed; -c 1000000 keeps seeds occurring up to 1e6 times
  # instead of discarding anything over 500. In alpha-satellite the default -c would
  # throw away precisely the seeds the alignment depends on. Do not drop these flags.
  # POST-ALIGNMENT FILTER: -F 3852 only.
  # The published description reads "trimmed and filtered (q20) BEFORE mapped ... and
  # samtools filtered (F3852)". So q20 is a pre-alignment READ-QUALITY filter (done by
  # fastp above), NOT a MAPQ filter. Adding `-q 20` here would drop multi-mapping reads
  # in alpha-satellite -- exactly what `-c 1000000` exists to preserve -- and would
  # silently gut the track. Set CENPA_MAPQ=20 only to test the alternative reading.
  bwa mem -t "$th" -k 50 -c 1000000 "$REF" "$WORK/${acc}_1.trim.fq.gz" "$WORK/${acc}_2.trim.fq.gz" \
       2>> "$WORK/$acc.bwa.log" \
    | samtools view -@ 2 -b ${CENPA_MAPQ:+-q $CENPA_MAPQ} -F 3852 - \
    | samtools sort -@ 3 -m 1G -T "$WORK/$acc.sorttmp" -o "$WORK/$acc.filt.bam" -
  samtools index -@ 2 "$WORK/$acc.filt.bam"
  rm -f "$WORK/${acc}_1.fastq.gz" "$WORK/${acc}_2.fastq.gz"   # keep the trimmed copies
}

make_bigwig () {                     # $1 = merged bam, $2 = output bw, $3 = threads
  if command -v fastp >/dev/null; then printf '  %-10s %s\n' fastp "$(command -v fastp)"
  elif [[ -x "$VENV/bin/cutadapt" ]]; then printf '  %-10s %s (fastp substitute)\n' cutadapt "$VENV/bin/cutadapt"
  else echo "  trimmer   MISSING -- run 'setup' or src/build_alignment_tools.sh"; ok=1; fi
  if [[ -x "$VENV/bin/bamCoverage" ]]; then
    "$VENV/bin/bamCoverage" -b "$1" -o "$2" -p "$3" --binSize 1000 \
        --normalizeUsing CPM --extendReads
  else
    echo "  using pure-python fallback (build_bigwig_from_bam.py)"
    "$VENV/bin/python" "$ROOT/src/build_bigwig_from_bam.py" "$1" "$2"
  fi
}

build_track () {                     # $1 = out.bw, $2 = threads, $3.. = accessions
  local out="$1" th="$2"; shift 2
  local bams=(); for a in "$@"; do align_one "$a" "$th"; bams+=("$WORK/$a.filt.bam"); done
  local merged="${out%.bw}.bam"
  if [[ ! -s "$merged" ]]; then
    if [[ "${#bams[@]}" -eq 1 ]]; then cp "${bams[0]}" "$merged"
    else samtools merge -@ "$th" -f "$merged" "${bams[@]}"; fi
    samtools index -@ 2 "$merged"
  fi
  make_bigwig "$merged" "$out" "$th"
  echo "  wrote $out"
}

cmd_run () {
  local th="${1:-8}"
  cmd_check || { echo "fix preflight first"; exit 1; }
  mkdir -p "$WORK"

  say "reference (chm13v2.0)"
  if [[ ! -s "$REF" ]]; then
    curl -L --no-progress-meter --retry 5 -C - -o "$WORK/chm13v2.0.fa.gz" \
      "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz"
    gunzip -c "$WORK/chm13v2.0.fa.gz" > "$REF"
  fi
  [[ -s "$REF.fai" ]] || samtools faidx "$REF"
  if [[ ! -s "$REF.bwt" ]]; then
    echo "  bwa index: single-threaded, 60-90 min on this machine. Leave it running."
    bwa index "$REF"
  fi

  say "CENP-A CUT&RUN (low-salt)"
  build_track "$OUT" "$th" "${RUNS[@]}"

  if [[ "${CENPA_CHIP:-0}" == "1" ]]; then
    say "CENP-A ChIP-seq (orthogonal check)"
    build_track "$WORK/chm13v2.0.chm13_CA_chipseq_q20.F3852.bw" "$th" "${CHIP[@]}"
  fi

  cat <<MSG

== re-run the analysis chain against the new track ==

  export CENPA_BW="$OUT"
  python src/build_human_arrays.py            # the 81-array analysis set
  python src/build_human_arrays_all.py        # all 114 arrays, no size filter
  python src/analyze_length_vs_homogeneity.py # the AUROC comparison
  python src/build_master_tables.py           # the per-chromosome master tables

build_human_arrays.py and build_human_arrays_all.py are the two that read CENPA_BW;
the other two work from the tables those write.
MSG
}

# ---------------------------------------------------------------- pilot ----
# `bwa mem -k 50 -c 1000000` is far more expensive than default bwa on satellite
# DNA, and CENP-A CUT&RUN reads are ENRICHED for alpha-satellite -- i.e. this data is
# the pathological case for -c 1000000. Rather than guess the full-run cost, measure
# it on a subsample and extrapolate. Run this before committing to the overnight job.
cmd_pilot () {
  local th="${1:-6}" pairs="${2:-2000000}"
  mkdir -p "$WORK"
  [[ -s "$REF.bwt" ]] || { echo "no bwa index yet -- run '$0 run' or build it first"; exit 1; }
  local acc="${RUNS[0]}" i=1
  for u in $(ena_urls "$acc"); do
    local f="$WORK/pilot_${i}.fq.gz"
    [[ -s "$f" ]] || { echo "  sampling $pairs pairs from $(basename "$u")"
      curl -sL "$u" | gunzip -c | head -n $(( pairs * 4 )) | gzip > "$f" || true; }
    i=$((i+1))
  done
  echo "  aligning $pairs pairs with the published flags, $th threads"
  /usr/bin/time -l bwa mem -t "$th" -k 50 -c 1000000 \
      "$REF" "$WORK/pilot_1.fq.gz" "$WORK/pilot_2.fq.gz" 2> "$WORK/pilot.time" \
    | samtools view -b ${CENPA_MAPQ:+-q $CENPA_MAPQ} -F 3852 - > "$WORK/pilot.bam"
  local secs peak
  secs=$(awk '/real/{print $1}' "$WORK/pilot.time" | tail -1)
  peak=$(awk '/maximum resident set size/{print $1}' "$WORK/pilot.time")
  [[ -z "$secs" ]] && secs=$(grep -oE '^[0-9.]+ real' "$WORK/pilot.time" | awk '{print $1}')
  echo
  echo "  pilot pairs      : $pairs"
  echo "  wall seconds     : ${secs:-see $WORK/pilot.time}"
  echo "  peak RSS         : $(( ${peak:-0} / 1073741824 )) GB  (index alone is ~4.5 GB)"
  if [[ -n "${secs:-}" ]]; then
    awk -v s="$secs" -v p="$pairs" 'BEGIN{
      tot=145656036; h=s*(tot/p)/3600;
      printf "  EXTRAPOLATED     : %.1f h for both runs (%d pairs total)\n", h, tot;
      if (h>18) print "  VERDICT          : too slow for a laptop -- rent a cloud VM, or use the published HG002 track";
      else if (h>8) print "  VERDICT          : feasible but long -- run it overnight, plugged in";
      else print "  VERDICT          : comfortably doable on this machine";
    }'
  fi
  echo "  (peak RSS above ~12 GB on 16 GB total means: lower the thread count)"
}

# ---------------------------------------------------------------- index ----
# Reference download + bwa index, split out so it can be staged on its own. The
# index build is single-threaded and takes 60-90 min; nothing else can start first.
cmd_index () {
  mkdir -p "$WORK"
  if [[ ! -s "$REF" ]]; then
    echo "  downloading chm13v2.0 (~1 GB gz)"
    curl -L --no-progress-meter --retry 5 -C - -o "$WORK/chm13v2.0.fa.gz" \
      "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz"
    gunzip -c "$WORK/chm13v2.0.fa.gz" > "$REF"
  fi
  [[ -s "$REF.fai" ]] || samtools faidx "$REF"
  if [[ ! -s "$REF.bwt" ]]; then
    echo "  bwa index: single-threaded, 60-90 min. Started $(date +%H:%M)."
    bwa index "$REF"
  fi
  echo "  index ready: $(ls -la "$REF.bwt" | awk '{print $5/1073741824" GB"}')"
}

case "${1:-run}" in
  setup) cmd_setup ;;
  index) cmd_index ;;
  check) cmd_check ;;
  pilot) cmd_pilot "${2:-6}" "${3:-2000000}" ;;
  run)   cmd_run "${2:-8}" ;;
  *)     echo "usage: $0 {setup|check|index|pilot [threads] [pairs]|run [threads]}"; exit 1 ;;
esac
