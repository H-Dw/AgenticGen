"""Collect a secret-free remote runtime inventory without installing packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

MODULES = (
    "torch",
    "torch_geometric",
    "torch_scatter",
    "rdkit",
    "pydantic",
    "httpx",
    "jsonschema",
    "numpy",
    "scipy",
    "pandas",
    "rdchiral",
    "syntheseus",
    "meeko",
    "gemmi",
    "openbabel",
    "plip",
    "openai",
    "litellm",
    "loguru",
    "networkx",
    "omegaconf",
    "yaml",
    "tqdm",
)


def collect() -> dict:
    packages = sorted(
        (
            {"name": item.metadata.get("Name"), "version": item.version}
            for item in importlib.metadata.distributions()
            if item.metadata.get("Name")
        ),
        key=lambda item: item["name"].lower(),
    )
    modules = {}
    for name in MODULES:
        try:
            spec = importlib.util.find_spec(name)
            modules[name] = {"available": spec is not None, "path": spec.origin if spec else None}
        except (ImportError, ValueError) as error:
            modules[name] = {"available": False, "error": str(error)}
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "modules": modules,
        "executables": {
            name: shutil.which(name)
            for name in (
                "conda",
                "mamba",
                "micromamba",
                "git",
                "vina",
                "plip",
                "obabel",
                "dot",
                "nvidia-smi",
                "rocm-smi",
                "hy-smi",
                "docker",
            )
        },
        "disk": {},
    }
    for path in ("/workspace", "/tmp", str(Path(sys.prefix).anchor)):
        if path and Path(path).exists():
            usage = shutil.disk_usage(path)
            result["disk"][path] = {"total": usage.total, "used": usage.used, "free": usage.free}
    if modules["torch"]["available"]:
        try:
            import torch

            result["torch_runtime"] = {
                "version": torch.__version__,
                "file": torch.__file__,
                "cuda_version": getattr(torch.version, "cuda", None),
                "hip_version": getattr(torch.version, "hip", None),
                "accelerator_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
        except Exception as error:
            result["torch_runtime"] = {"error": repr(error)}
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    result["pip"] = {"returncode": pip.returncode, "version": pip.stdout.strip()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
