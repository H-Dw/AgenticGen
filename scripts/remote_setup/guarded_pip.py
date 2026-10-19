"""Install extra packages while refusing changes to the existing accelerator stack.

Run with the *target environment's* Python. The pip resolution is inspected before
installation; installed accelerator and numerical package versions are constrained.
No credentials, environment variables, or authenticated download URLs are logged by
this helper. Use only public package specifications, not authenticated URLs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROTECTED_EXACT = {
    "numpy",
    "scipy",
    "rdkit",
    "pytorch-lightning",
    "lightning",
    "dgl",
    "pyg-lib",
    "triton",
    "dtk",
    "torch",
    "torchvision",
    "torchaudio",
}
PROTECTED_PREFIXES = ("torch-", "pytorch-", "nvidia-", "musa-", "ditorch", "dtk-")
CAN_ADD_IF_MISSING = {"numpy", "scipy", "rdkit"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def protected(name: str) -> bool:
    return name in PROTECTED_EXACT or name.startswith(PROTECTED_PREFIXES)


def inventory() -> dict[str, str]:
    # A runtime .pth can expose both environment and image distributions. Match
    # importlib.metadata.version's first-match precedence rather than overwriting
    # the active environment's version with a later system-image distribution.
    result = {}
    for item in importlib.metadata.distributions():
        name = item.metadata.get("Name")
        if name and protected(normalize(name)):
            result.setdefault(normalize(name), item.version)
    return result


def outside_prefix(name: str) -> bool:
    """A .pth may expose image packages that this installer must not uninstall."""
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    location = Path(distribution.locate_file("")).resolve()
    return not location.is_relative_to(Path(sys.prefix).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-deps", action="store_true")
    parser.add_argument("--no-build-isolation", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Otherwise resolve only")
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args()
    before = inventory()
    with tempfile.TemporaryDirectory(prefix="agenticgen-pip-") as temporary:
        root = Path(temporary)
        constraints = root / "protected.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in sorted(before.items())),
            encoding="utf-8",
        )
        report_path = root / "resolution.json"
        command = [sys.executable, "-m", "pip", "install", "--constraint", str(constraints)]
        if args.no_deps:
            command.append("--no-deps")
        if args.no_build_isolation:
            command.append("--no-build-isolation")
        command.extend(args.packages)
        subprocess.run(
            [*command, "--dry-run", "--report", str(report_path)],
            check=True,
        )
        resolution = json.loads(report_path.read_text(encoding="utf-8"))
        planned = [
            {"name": item["metadata"]["name"], "version": item["metadata"]["version"]}
            for item in resolution.get("install", [])
        ]
        forbidden = [
            item
            for item in planned
            if protected(normalize(item["name"]))
            and (
                normalize(item["name"]) in before
                or normalize(item["name"]) not in CAN_ADD_IF_MISSING
            )
        ]
        if forbidden:
            raise SystemExit(
                "Refusing accelerator/numerical stack changes; use a separate CPU worker: "
                + json.dumps(forbidden)
            )
        external_changes = [item for item in planned if outside_prefix(item["name"])]
        if external_changes:
            raise SystemExit(
                "Refusing changes to packages imported from outside this environment; "
                "isolate installation or use a separate CPU worker: " + json.dumps(external_changes)
            )
        print(json.dumps({"protected_before": before, "planned": planned}, indent=2))
        if args.execute:
            subprocess.run(command, check=True)
            after = inventory()
            existing_changed = any(after.get(name) != version for name, version in before.items())
            unexpected_added = (after.keys() - before.keys()) - CAN_ADD_IF_MISSING
            if existing_changed or unexpected_added:
                raise SystemExit(
                    "Protected package inventory changed; stop and inspect environment"
                )
            print("Protected package inventory unchanged.")


if __name__ == "__main__":
    main()
