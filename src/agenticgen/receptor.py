"""Prepare the local CK2 receptor and a reference-ligand box without downloading structures."""

import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem

from agenticgen.io import sha256, write_json


def prepare_receptor(
    protein: Path,
    ligand: Path,
    output: Path,
    image: str,
    padding: float = 8.0,
    minimum_size: float = 22.5,
) -> dict:
    if padding <= 0 or minimum_size <= 0:
        raise ValueError("Docking box dimensions must be positive")
    molecules = list(Chem.SDMolSupplier(str(ligand), removeHs=True))
    if len(molecules) != 1 or molecules[0] is None or not molecules[0].GetNumConformers():
        raise ValueError("Provide one reference ligand in the receptor coordinate frame")
    coords = molecules[0].GetConformer().GetPositions()
    if not np.isfinite(coords).all():
        raise ValueError("Reference coordinates must be finite")
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    center = (lower + upper) / 2
    size = np.maximum(upper - lower + 2 * padding, minimum_size)
    lines = [
        line
        for line in protein.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM  ")
    ]
    if not lines:
        raise ValueError("Receptor has no protein ATOM records")
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    output.mkdir(parents=True, exist_ok=False)
    pdb, pdbqt, box = output / "receptor.pdb", output / "receptor.pdbqt", output / "vina.cfg"
    pdb.write_text("\n".join([*lines, "END"]) + "\n", encoding="utf-8")
    box.write_text(
        "".join(f"center_{axis} = {value:.6f}\n" for axis, value in zip("xyz", center, strict=True))
        + "".join(f"size_{axis} = {value:.6f}\n" for axis, value in zip("xyz", size, strict=True))
        + "num_modes = 9\n",
        encoding="utf-8",
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        f"type=bind,source={output.resolve()},target=/job",
        inspect.stdout.strip(),
        "obabel",
        "/job/receptor.pdb",
        "-O",
        "/job/receptor.pdbqt",
        "-xr",
    ]
    manifest = {
        "schema_version": "agenticgen.receptor.v1",
        "target": "CK2",
        "status": "running",
        "inputs": {"protein_sha256": sha256(protein), "reference_ligand_sha256": sha256(ligand)},
        "protein_atom_count": len(lines),
        "image_id": inspect.stdout.strip(),
        "preparation": "upstream skill OpenBabel rigid-receptor conversion; ATOM only; no protonation-state search",
        "box_policy": "reference-ligand bounding box plus padding, minimum size per axis",
        "padding_A": padding,
        "center_A": center.tolist(),
        "size_A": size.tolist(),
        "interpretation": "Protein and reference ligand must already share a coordinate frame",
    }
    write_json(output / "manifest.json", manifest)
    try:
        with (output / "preparation.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        if not pdbqt.is_file() or "ATOM" not in pdbqt.read_text(encoding="utf-8"):
            raise ValueError("OpenBabel produced no receptor PDBQT atoms")
        manifest.update(status="prepared", outputs={p.name: sha256(p) for p in (pdb, pdbqt, box)})
        return manifest
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "failed"
        write_json(output / "manifest.json", manifest)
