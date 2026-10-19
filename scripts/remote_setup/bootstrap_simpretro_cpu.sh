#!/usr/bin/env bash
# Match the existing SimpRetro Docker recipe in a separate, non-GPU environment.
set -euo pipefail

prefix="/workspace/envs/agenticgen-simpretro"
repo="/workspace/AgenticGen"
mamba_bin="/opt/miniforge3/bin/mamba"
audit_dir="/workspace/agenticgen-environment"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
explicit_lock="$script_dir/simpretro-linux-64.explicit.txt"
while (($#)); do
  case "$1" in
    --prefix) prefix="${2:?missing worker prefix}"; shift 2 ;;
    --repo) repo="${2:?missing repository}"; shift 2 ;;
    --mamba) mamba_bin="${2:?missing mamba executable}"; shift 2 ;;
    --audit-dir) audit_dir="${2:?missing audit directory}"; shift 2 ;;
    --explicit-lock) explicit_lock="${2:?missing lock file}"; shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! -x "$mamba_bin" || ! -f "$repo/environments/simpretro-worker/environment.yml" ]]; then
  printf 'Supply existing mamba and AgenticGen checkout paths.\n' >&2
  exit 2
fi
mkdir -p -- "$audit_dir"
if [[ ! -x "$prefix/bin/python" ]]; then
  if [[ -f "$explicit_lock" ]]; then
    "$mamba_bin" create --yes --prefix "$prefix" --file "$explicit_lock"
  else
    # The recipe combines pytorch and conda-forge. Strict priority would hide
    # required conda-forge runtime libraries behind old pytorch-channel builds.
    "$mamba_bin" env create --channel-priority flexible --yes --prefix "$prefix" \
      --file "$repo/environments/simpretro-worker/environment.yml"
  fi
fi
"$prefix/bin/python" -m pip install --no-deps --no-build-isolation \
  -e "$repo/external/SimpRetro"
"$mamba_bin" list --prefix "$prefix" --explicit > "$audit_dir/cpu-worker-explicit.txt"
"$prefix/bin/python" - <<'PY'
import json
import sys
import torch
from rdkit import rdBase
from rdchiral.main import rdchiralRunText

assert sys.version_info[:2] == (3, 10), sys.version
assert str(torch.__version__).startswith("2.5.1"), torch.__version__
assert torch.version.cuda is None, "SimpRetro worker must remain CPU-only"
assert rdBase.rdkitVersion == "2024.09.6", rdBase.rdkitVersion
assert rdchiralRunText("[C:1](=[O:2])[O:3]>>[C:1](=[O:2])Cl.[O:3]", "CC(=O)O")
print(json.dumps({"python": sys.executable, "version": sys.version,
                  "torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "rdkit": rdBase.rdkitVersion, "rdchiral_template_smoke": True}))
PY
printf 'CPU worker ready: %s/bin/python\n' "$prefix"
