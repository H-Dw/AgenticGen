"""Measure a supplied pose on a copy; never replace its coordinates."""

import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from agenticgen.io import sha256


def measure_pose(path: Path) -> dict:
    supplier = Chem.SDMolSupplier(str(path), removeHs=True)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError("Pose quality requires one readable molecule")
    mol = supplier[0]
    if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError("Pose lacks 3D coordinates")
    coords = np.asarray(mol.GetConformer().GetPositions())
    if not np.isfinite(coords).all():
        raise ValueError("Pose has non-finite coordinates")
    table = Chem.GetPeriodicTable()
    long_or_short = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        length = float(np.linalg.norm(coords[a.GetIdx()] - coords[b.GetIdx()]))
        radius_sum = table.GetRcovalent(a.GetAtomicNum()) + table.GetRcovalent(b.GetAtomicNum())
        if length < 0.6 * radius_sum or length > 1.5 * radius_sum:
            long_or_short.append(
                {
                    "atom_maps": [a.GetAtomMapNum(), b.GetAtomMapNum()],
                    "length_angstrom": length,
                }
            )
    energy = {"status": "unavailable", "method": "MMFF94s same-molecule relaxation"}
    copy = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    if AllChem.MMFFHasAllMoleculeParams(copy):
        properties = AllChem.MMFFGetMoleculeProperties(copy, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(copy, properties)
        before = ff.CalcEnergy()
        code = ff.Minimize(maxIts=1000)
        after = ff.CalcEnergy()
        if all(math.isfinite(v) for v in (before, after)):
            energy.update(
                status="ok",
                raw_energy_kcal_mol=before,
                relaxed_energy_kcal_mol=after,
                relief_kcal_mol=before - after,
                relief_per_heavy_atom=(before - after) / mol.GetNumAtoms(),
                minimization_converged=code == 0,
            )
    return {
        "input_pose_sha256": sha256(path),
        "heavy_atom_count": mol.GetNumAtoms(),
        "extreme_bond_lengths": long_or_short,
        "energy": energy,
        "interpretation": "Energy relief compares one graph before/after relaxation; absolute energies are not comparable across graphs. These checks do not certify physical validity.",
    }
