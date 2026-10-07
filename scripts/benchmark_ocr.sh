#!/usr/bin/env bash
# benchmark_ocr.sh — run the ocr-core quality harness (CER/WER) from one command.
#
# Clean-room note: this script orchestrates ocr-core's own benchmarks; it
# borrows only the *concept* of standardized quality measurement from
# publicly documented commercial-tool behaviour (see docs/research/abbyy_features.md).
#
# Usage:
#   scripts/benchmark_ocr.sh                          # golden suite smoke run (limit 20)
#   scripts/benchmark_ocr.sh --full                   # full golden suite
#   scripts/benchmark_ocr.sh --engines tesseract      # restrict engines
#   scripts/benchmark_ocr.sh --out /tmp/bench-report  # custom report dir
set -euo pipefail

cd "$(dirname "$0")/.."

EXTRA_ARGS=()
OUT_DIR="reports/golden-$(date +%Y%m%d-%H%M%S)"
FULL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --engines) EXTRA_ARGS+=(--engines "$2"); shift 2 ;;
    --lang) EXTRA_ARGS+=(--lang "$2"); shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $FULL -eq 1 ]]; then
  :
else
  EXTRA_ARGS+=(--limit 20)
fi

python3 -m ocr_core.benchmarks.golden --out "$OUT_DIR" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
echo "report: $OUT_DIR/report.md"
