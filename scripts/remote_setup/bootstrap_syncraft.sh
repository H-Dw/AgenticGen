#!/usr/bin/env bash
# Add SynCraft tools without replacing an existing FLOWR/DTK accelerator stack.
set -euo pipefail

prefix=""
install_chemistry=0
install_system=0
audit_dir="/workspace/agenticgen-environment"
while (($#)); do
  case "$1" in
    --prefix) prefix="${2:?missing environment prefix}"; shift 2 ;;
    --audit-dir) audit_dir="${2:?missing audit directory}"; shift 2 ;;
    --with-chemistry) install_chemistry=1; shift ;;
    --with-system-tools) install_system=1; shift ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$prefix" || ! -x "$prefix/bin/python" ]]; then
  printf 'Supply --prefix pointing to the existing molsteer-flowr-dtk environment.\n' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$prefix/bin/python"
export PATH="$prefix/bin:$PATH"
mkdir -p -- "$audit_dir"
"$python_bin" "$script_dir/inspect_environment.py" --output "$audit_dir/before.json"

# Resolve first, reject protected/external-prefix changes, then execute that plan.
"$python_bin" "$script_dir/guarded_pip.py" --execute \
  'pydantic>=2.8,<3' 'httpx>=0.27' 'jsonschema>=4.21' \
  'loguru>=0.7' 'tqdm>=4.66' 'openai>=1' 'litellm==1.53.1' \
  'more-itertools>=10' 'networkx>=3' 'omegaconf>=2.3' 'pyyaml>=6' \
  'pandas>=2' 'graphviz==0.20.3' rdchiral

if ((install_chemistry)); then
  # Do not install a second OpenBabel distribution over an existing conda package.
  if ! "$python_bin" -c 'from openbabel import openbabel; assert openbabel.OBConversion()' ; then
    "$python_bin" "$script_dir/guarded_pip.py" --execute 'openbabel-wheel==3.1.1.22'
  fi
  "$python_bin" "$script_dir/guarded_pip.py" --execute \
    'meeko==0.7.1' 'gemmi>=0.7.3,<0.8' 'biopython>=1.85' 'lxml>=5.4'
  # PLIP's pip metadata requests a source OpenBabel package; use the tested binding
  # above rather than compiling/replacing OpenBabel as a transitive dependency.
  "$python_bin" "$script_dir/guarded_pip.py" --execute --no-deps \
    --no-build-isolation 'plip==3.0.0'
fi

if ((install_system)); then
  if ! command -v apt-get >/dev/null; then
    printf 'System tools require an apt-based host; install Vina/Graphviz manually.\n' >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-remove --no-install-recommends \
    autodock-vina graphviz libxrender1 libxext6 libsm6
fi

"$python_bin" "$script_dir/inspect_environment.py" --output "$audit_dir/after.json"
if ((install_chemistry)); then
  "$python_bin" "$script_dir/smoke_tools.py" --output "$audit_dir/tools-smoke.json"
fi
printf 'Extra dependency setup finished; inspect %s before running model examples.\n' "$audit_dir"
