"""Run deterministic chemistry/tool checks; no model inference or external APIs."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-retro", action="store_true")
    args = parser.parse_args()
    checks = {}
    modules = ["rdkit", "pydantic", "httpx", "jsonschema", "meeko", "gemmi", "openbabel", "plip"]
    if args.require_retro:
        modules.extend(["rdchiral", "syntheseus", "torch", "omegaconf", "networkx"])
    for name in modules:
        try:
            module = importlib.import_module(name)
            checks[name] = {"ok": True, "path": str(getattr(module, "__file__", None))}
            try:
                checks[name]["version"] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                checks[name]["version"] = None
        except Exception as error:
            checks[name] = {"ok": False, "error": repr(error)}
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from openbabel import openbabel
        from rdkit import Chem
        from rdkit.Chem import AllChem

        molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
        preparation = MoleculePreparation()
        setups = preparation.prepare(molecule)
        pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
        assert success and "ROOT" in pdbqt, error
        conversion = openbabel.OBConversion()
        assert conversion.SetInAndOutFormats("pdb", "pdbqt")
        checks["ligand_preparation"] = {"ok": True, "seed": 42}
        with tempfile.TemporaryDirectory(prefix="agenticgen-meeko-") as temporary:
            legacy_path = Path(temporary) / "ligand.pdbqt"
            assert callable(getattr(preparation, "write_pdbqt_file", None))
            preparation.write_pdbqt_file(str(legacy_path))
            assert "ROOT" in legacy_path.read_text(encoding="utf-8")
            checks["syncraft_legacy_meeko_writer"] = {"ok": True}
    except Exception as error:
        checks["ligand_preparation"] = {"ok": False, "error": repr(error)}
    if args.require_retro:
        try:
            from rdchiral.main import rdchiralRunText

            outputs = rdchiralRunText("[C:1](=[O:2])[O:3]>>[C:1](=[O:2])Cl.[O:3]", "CC(=O)O")
            assert outputs
            checks["rdchiral_template_execution"] = {"ok": True, "outputs": outputs}
        except Exception as error:
            checks["rdchiral_template_execution"] = {"ok": False, "error": repr(error)}
    for binary, argument in (("vina", "--version"), ("plip", "--help"), ("obabel", "-V")):
        executable = shutil.which(binary)
        if not executable:
            checks[f"binary_{binary}"] = {"ok": False, "error": "not found on PATH"}
            continue
        proc = subprocess.run(
            [executable, argument],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        checks[f"binary_{binary}"] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "output": (proc.stdout + proc.stderr)[-2000:],
        }
    result = {"ok": all(item["ok"] for item in checks.values()), "checks": checks}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
