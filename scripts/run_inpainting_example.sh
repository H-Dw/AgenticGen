#!/usr/bin/env bash
set -euo pipefail

kind="${1:-replay}"
destination="${2:?Usage: bash scripts/run_inpainting_example.sh replay|live NEW_OUTPUT_DIR [retro]}"
retro="${3:-}"
if [[ "$kind" != replay && "$kind" != live ]]; then
  printf '%s\n' 'First argument must be replay or live' >&2
  exit 2
fi
if [[ -e "$destination" ]]; then
  printf '%s\n' 'Use a new output directory to preserve previous results' >&2
  exit 2
fi
export AGENTICGEN_PYTHON="${AGENTICGEN_PYTHON:-$(command -v python)}"
export FLOWR_PYTHON="${FLOWR_PYTHON:-$AGENTICGEN_PYTHON}"
export VINA_BINARY="${VINA_BINARY:-$(command -v vina)}"
: "${FLOWR_CHECKPOINT:?Set FLOWR_CHECKPOINT to the released pocket-model checkpoint}"
evaluation=examples/ck2_inpainting/evaluation.json
if [[ "$retro" == retro ]]; then
  : "${SIMP_RETRO_PYTHON:?Set the independent Python 3.10 SimpRetro worker interpreter}"
  : "${SYNCRAFT_STOCK:?Set the official inventory path}"
  : "${SYNCRAFT_STOCK_INDEX:?Set its verified SQLite index path}"
  : "${SYNCRAFT_FINGERPRINT_CACHE:?Set the verified template cache path}"
  evaluation=examples/ck2_inpainting/evaluation.retro.json
fi
if [[ "$kind" == live ]]; then
  : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the process environment}"
fi
mkdir -p "$destination"
"$AGENTICGEN_PYTHON" -m agenticgen design \
  --config "examples/ck2_inpainting/design.$kind.json" --output "$destination/design"
"$AGENTICGEN_PYTHON" -m agenticgen inpaint \
  --run "$destination/design" --config examples/ck2_inpainting/flowr.json \
  --pocket-pdb examples/ck2_inpainting/inputs/receptor.pdb \
  --output "$destination/inpainting" --execute --evaluation-config "$evaluation"
