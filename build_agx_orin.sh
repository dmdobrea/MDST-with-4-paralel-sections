#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-mdst_batched_agx_orin.cu}"
EXE="${2:-mdst_batched_agx_orin}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found in PATH." >&2
  exit 1
fi

echo "Building for Jetson AGX Orin (Ampere, compute capability 8.7)..."
# -lineinfo helps Nsight Compute associate metrics with source lines.
# -Xptxas=-v prints register/shared-memory usage at compile time.
nvcc -O3 -std=c++17 -arch=sm_87 -lineinfo -Xptxas=-v \
  "$SRC" -o "$EXE" 2> ptxas_report.txt

echo "Built: $EXE"
echo "PTXAS resource report: ptxas_report.txt"
grep -E "ptxas info.*(Used|Compiling)" ptxas_report.txt || true
