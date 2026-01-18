#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./bin/cublas_profile}
SIZE=${2:-1024}
ITERS=${3:-10}
OUT_DIR=${4:-profile/out}
KERNEL_REGEX=${5:-.*sgemm.*}

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu (Nsight Compute) not found in PATH." >&2
  echo "Install Nsight Compute and ensure 'ncu' is available." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Detailed memory + scheduler stats to infer memory latency bottlenecks.
SECTIONS=(
  MemoryWorkloadAnalysis
  SchedulerStats
  WarpState
  SpeedOfLight
  LaunchStats
)

SECTION_ARGS=()
for s in "${SECTIONS[@]}"; do
  SECTION_ARGS+=("--section" "$s")
done

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$OUT_DIR/ncu_${SIZE}_${ITERS}_$STAMP"

ncu \
  --target-processes all \
  --kernel-name "regex:${KERNEL_REGEX}" \
  --import-source yes \
  "${SECTION_ARGS[@]}" \
  --export "${OUT_BASE}.ncu-rep" \
  --force-overwrite \
  "$BIN" "$SIZE" "$ITERS"

echo "Report: ${OUT_BASE}.ncu-rep"
