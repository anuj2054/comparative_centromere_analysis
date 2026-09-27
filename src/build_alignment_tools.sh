#!/usr/bin/env bash
# Build the CENP-A alignment toolchain into ~/.local without admin rights.
# Homebrew needs sudo (this account has no passwordless sudo), but clang/make/git are
# present via the Xcode command line tools, so bwa and samtools compile from source.
#
# fastp is replaced by cutadapt: fastp needs libdeflate/isa-l and is painful to build
# here, while cutadapt is pip-installable and does the same job for this recipe
# ("trimmed and filtered (q20)" -- neither tool is named in the published methods).
set -euo pipefail
PREFIX="$HOME/.local"
SRC="${CENPA_WORK:-$(cd "$(dirname "$0")/.." && pwd)/work/cenpa}/toolsrc"
VENV="${CENPA_VENV:-$(cd "$(dirname "$0")/.." && pwd)/work/cenpa/venv}"
mkdir -p "$PREFIX/bin" "$SRC"
cd "$SRC"

echo "== bwa =="
if [[ ! -x "$PREFIX/bin/bwa" ]]; then
  [[ -d bwa ]] || git clone --depth 1 https://github.com/lh3/bwa.git
  cd bwa
  if ! make -j4 2>build1.log; then
    echo "  plain build failed (expected on arm64: SSE2 intrinsics). Patching with sse2neon."
    make clean >/dev/null 2>&1 || true
    curl -fsSLO https://raw.githubusercontent.com/DLTcollab/sse2neon/master/sse2neon.h
    for f in *.c; do
      /usr/bin/sed -i '' 's|#include <emmintrin.h>|#include "sse2neon.h"|' "$f" || true
    done
    /usr/bin/sed -i '' 's/-msse2//g' Makefile || true
    make -j4 CFLAGS="-g -Wall -Wno-unused-function -Wno-unused-variable -O2 -fcommon" 2>build2.log
  fi
  cp bwa "$PREFIX/bin/"; cd "$SRC"
fi
"$PREFIX/bin/bwa" 2>&1 | head -3 || true

echo "== htslib + samtools =="
if [[ ! -x "$PREFIX/bin/samtools" ]]; then
  V=1.21
  [[ -d "samtools-$V" ]] || { curl -fsSLO "https://github.com/samtools/samtools/releases/download/$V/samtools-$V.tar.bz2"; tar xf "samtools-$V.tar.bz2"; }
  cd "samtools-$V"
  ./configure --prefix="$PREFIX" --without-curses --disable-lzma --disable-bz2 >conf.log 2>&1
  make -j6 >make.log 2>&1
  make install >install.log 2>&1
  cd "$SRC"
fi
"$PREFIX/bin/samtools" --version | head -2

echo "== python side (cutadapt, deeptools, pysam, pyBigWig) =="
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q cutadapt pysam pyBigWig numpy
"$VENV/bin/pip" install -q deeptools 2>/dev/null || echo "  deeptools unavailable -- build_bigwig_from_bam.py fallback will be used"
"$VENV/bin/cutadapt" --version | sed 's/^/  cutadapt /'

echo
echo "toolchain ready. Add to PATH:  export PATH=\"$PREFIX/bin:\$PATH\""
