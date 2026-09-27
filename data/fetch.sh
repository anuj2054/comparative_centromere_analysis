#!/usr/bin/env bash
#
# Fetch the reference inputs. Nothing under data/ is redistributed in this
# repository: the assemblies, annotations and reads all belong to their original
# projects and are downloaded from the primary sources here.
#
#   ./data/fetch.sh assemblies     human + three great-ape assemblies (~4 GB)
#   ./data/fetch.sh annotations    CenSat tracks, Fiber-seq, ONT methylation
#   ./data/fetch.sh extracts       re-fetch the four small chr21 FASTAs (~5 MB)
#   ./data/fetch.sh reads          CENP-A / IgG / ChIP-seq runs (~200 GB)
#   ./data/fetch.sh all
#
# `extracts` is the cheap one, a few seconds, and `make figures` runs it for you.
# The four are not committed: nothing under data/ is. Set FORCE=1 to re-fetch copies
# already present, after changing a region in src/fetch_chr21_extracts.py.
#
# After fetching, verify with:  shasum -a 256 -c data/checksums.sha256
#
set -euo pipefail
cd "$(dirname "$0")"

T2T=https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies
ARK=https://s3.amazonaws.com/genomeark/species

get () {  # get <url> <filename>
  [ -s "$2" ] && { echo "  have $2"; return 0; }
  echo "  fetching $2"
  curl -L --fail --retry 8 --retry-delay 15 --retry-all-errors -o "$2.part" "$1"
  mv "$2.part" "$2"
}

assemblies () {
  echo "assemblies:"
  # Human. GCA_009914755.4 / T2T-CHM13 v2.0. The atlas uses all 24 chromosomes,
  # NC_060925.1 through NC_060948.1.
  get "$T2T/analysis_set/chm13v2.0.fa.gz" chm13v2.0.fa.gz
  # Great apes, primary haplotypes. Alternate haplotypes are the replication set;
  # swap hap1 -> hap2 and mat -> pat to fetch those.
  get "$ARK/Pan_troglodytes/mPanTro3/assembly_curated/mPanTro3.hap1.cur.20231031.fasta.gz" mPanTro3.hap1.cur.20231031.fasta.gz
  get "$ARK/Gorilla_gorilla/mGorGor1/assembly_curated/mGorGor1.mat.cur.20231031.fasta.gz" mGorGor1.mat.cur.20231031.fasta.gz
  get "$ARK/Pan_paniscus/mPanPan1/assembly_curated/mPanPan1.mat.cur.20231031.fasta.gz" mPanPan1.mat.cur.20231031.fasta.gz
}

annotations () {
  echo "annotations:"
  # CenSat v2.1 for human; v1.2 for the apes. These define the candidate arrays,
  # and the active_hor / dhor distinction is the entire task definition.
  get "$T2T/annotation/chm13v2.0_censat_v2.1.bed" chm13v2.0_censat_v2.1.bed
  cat <<'MSG'
  The ape CenSat bigBeds and the two CHM13 signal tracks are distributed through
  the T2T and GenomeArk browser hubs rather than as stable flat-file URLs, so they
  are listed here by exact filename instead of being fetched blindly:

    mPanTro3_v2.0_CenSat_v1.2.bb
    mGorGor1_v2.0_CenSat_v1.2.bb
    mPanPan1_v2.0_CenSat_v1.2.bb
    all.percent.accessible.bw          CHM13 Fiber-seq, NOT an HG002 track
    chm13v2.0_nanopore_CpG.bw          CHM13 ONT CpG methylation

  Place them in data/ and re-run data/make_checksums.sh.
MSG
}

extracts () {
  echo "extracts:"
  # The three chr21 regions and the twelve RefSeq CDS that make the coding class.
  # src/fetch_chr21_extracts.py holds the coordinates and the accession list, and writes
  # manifest.json alongside; it uses only the standard library, so this runs before
  # pip install. Duplicating the regions here would let the two drift apart.
  local want=(chr21_map.fasta satellite.fasta unique.fasta coding_cds.fasta manifest.json)
  if [ "${FORCE:-0}" != "1" ]; then
    local missing=0
    for f in "${want[@]}"; do [ -s "$f" ] || missing=1; done
    [ "$missing" -eq 0 ] && {
      echo "  have all four extracts and manifest.json (FORCE=1 to re-fetch)"
      echo "  verify them with: shasum -a 256 --ignore-missing -c checksums.sha256"
      return 0
    }
  fi
  "${PY:-python3}" ../src/fetch_chr21_extracts.py
}

reads () {
  echo "reads: BioProject PRJNA559484 (CUT&RUN) and the matched ChIP-seq runs"
  command -v fasterq-dump >/dev/null || { echo "  needs sra-tools (fasterq-dump)"; return 1; }
  # low-salt CENP-A CUT&RUN (the reported primary label) and its matched IgG
  for s in SRR15395851 SRR15395853 SRR15395849 SRR15395847; do
    [ -s "${s}_1.fastq.gz" ] || fasterq-dump --split-files --progress "$s"
  done
  # high-salt CENP-A pair, aligned at full depth so salt and depth are not confounded
  for s in SRR15395852 SRR15395850; do
    [ -s "${s}_1.fastq.gz" ] || fasterq-dump --split-files --progress "$s"
  done
  # orthogonal CENP-A ChIP-seq with its own matched input
  for s in SRR13278683 SRR13278684 SRR13278681 SRR13278682; do
    [ -s "${s}_1.fastq.gz" ] || fasterq-dump --split-files --progress "$s"
  done
  cat <<'MSG'
  Alignment recipe (Methods 2.4): q20 prefilter, then
    bwa mem -k 50 -c 1000000       repeat-aware, retains multi-mapping alpha-satellite
    samtools view -F 3852
  No post-alignment MAPQ threshold is applied. A MAPQ filter here would discard the
  multi-mapping reads the permissive settings exist to keep, and would silently
  deplete exactly the arrays under study.
MSG
}

# Subcommands may be combined: `fetch.sh assemblies annotations` runs both, in the
# order given.
for arg in "${@:-all}"; do
  case "$arg" in
    assemblies) assemblies ;;
    annotations) annotations ;;
    extracts) extracts ;;
    reads) reads ;;
    all) assemblies; annotations; extracts; reads ;;
    *) sed -n '2,19p' "$0"; exit 1 ;;
  esac
done
