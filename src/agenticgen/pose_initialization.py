"""Initialize an edited graph in the source pocket frame, before FLOWR sampling.

These coordinates are initial conditions, not FLOWR predictions or evidence of a
reasonable bound pose. Existing heavy-atom positions are copied exactly by map ID.
"""

from __future__ import annotations

import math
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from agenticgen.chemistry import canonical_smiles, molecule
from agenticgen.contracts import GraphEditResult
from agenticgen.io import sha256, write_json


def _atoms(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    atoms = {a.GetAtomMapNum(): a for a in mol.GetAtoms()}
    if len(atoms) != mol.GetNumAtoms() or 0 in atoms:
        raise ValueError("Every heavy atom must have a unique positive atom-map ID")
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError("The initialization contract uses mapped heavy atoms only")
    return atoms


def _check_mapped_graph(expected: Chem.Mol, actual: Chem.Mol) -> None:
    """Validate a graph and its correspondence, allowing unspecified stereo in SMILES."""
    expected_atoms, actual_atoms = _atoms(expected), _atoms(actual)
    if set(expected_atoms) != set(actual_atoms):
        raise ValueError("Source pose atom maps differ from the expected source graph")
    for map_id, expected_atom in expected_atoms.items():
        actual_atom = actual_atoms[map_id]
        if (
            expected_atom.GetAtomicNum(),
            expected_atom.GetFormalCharge(),
            expected_atom.GetIsotope(),
        ) != (actual_atom.GetAtomicNum(), actual_atom.GetFormalCharge(), actual_atom.GetIsotope()):
            raise ValueError(f"Pose atom identity differs at atom map {map_id}")
    expected_bonds = {
        tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))): str(
            b.GetBondType()
        )
        for b in expected.GetBonds()
    }
    actual_bonds = {
        tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))): str(
            b.GetBondType()
        )
        for b in actual.GetBonds()
    }
    if expected_bonds != actual_bonds:
        raise ValueError("Pose mapped connectivity differs from its expected graph")
    # A full graph match checks specified stereochemistry. We also check the mapped
    # ordering so a symmetry-related alternative cannot silently remap a contact.
    order = tuple(actual_atoms[a.GetAtomMapNum()].GetIdx() for a in expected.GetAtoms())
    matches = actual.GetSubstructMatches(
        expected, useChirality=True, uniquify=False, maxMatches=100000
    )
    if order not in matches:
        raise ValueError("Pose contradicts the specified source/product stereochemistry")


def _coordinate_copy(product, product_atoms, source_positions):
    conf = Chem.Conformer(product.GetNumAtoms())
    conf.Set3D(True)
    for map_id, atom in product_atoms.items():
        conf.SetAtomPosition(atom.GetIdx(), source_positions[map_id])
    product.RemoveAllConformers()
    product.AddConformer(conf)
    return product


def _initialize_new_atoms(product, product_atoms, source_positions, seed, metadata):
    """Try constrained ETKDG once, then relax only new atoms and hydrogens."""
    shared = set(product_atoms) & set(source_positions)
    if not shared:
        raise ValueError("No surviving source atoms anchor new atoms to the pocket frame")
    working = Chem.AddHs(product)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.numThreads = 1
    params.SetCoordMap({product_atoms[i].GetIdx(): source_positions[i] for i in shared})
    if AllChem.EmbedMolecule(working, params) != 0:
        raise ValueError("Constrained ETKDG could not initialize the added atoms")
    conf = working.GetConformer()
    for map_id in shared:
        conf.SetAtomPosition(product_atoms[map_id].GetIdx(), source_positions[map_id])
    if AllChem.MMFFHasAllMoleculeParams(working):
        properties = AllChem.MMFFGetMoleculeProperties(working)
        forcefield = AllChem.MMFFGetMoleculeForceField(working, properties)
        forcefield_name = "MMFF94"
    elif AllChem.UFFHasAllMoleculeParams(working):
        forcefield = AllChem.UFFGetMoleculeForceField(working)
        forcefield_name = "UFF"
    else:
        raise ValueError("No complete force-field parameters for initializing the added atoms")
    if forcefield is None:
        raise ValueError("Could not construct a constrained initialization force field")
    for map_id in shared:
        forcefield.AddFixedPoint(product_atoms[map_id].GetIdx())
    forcefield.Initialize()
    convergence = forcefield.Minimize(maxIts=1000)
    metadata.update(initialization_forcefield=forcefield_name, minimizer_status=convergence)
    if convergence != 0:
        raise ValueError("Constrained new-atom initialization did not converge")
    # Fixed-point constraints are verified and copied exactly once more before SDF
    # serialization. This is coordinate initialization, never an edited-graph repair.
    for map_id in shared:
        conf.SetAtomPosition(product_atoms[map_id].GetIdx(), source_positions[map_id])
    result = Chem.RemoveHs(working)
    positions = result.GetConformer()
    new_maps = set(product_atoms) - shared
    table = Chem.GetPeriodicTable()
    for bond in result.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetAtomMapNum() not in new_maps and right.GetAtomMapNum() not in new_maps:
            continue
        separation = (
            positions.GetAtomPosition(left.GetIdx()) - positions.GetAtomPosition(right.GetIdx())
        ).Length()
        expected = table.GetRcovalent(left.GetAtomicNum()) + table.GetRcovalent(
            right.GetAtomicNum()
        )
        if not (0.55 * expected <= separation <= 1.55 * expected):
            raise ValueError("Initialized added-atom bond has implausible length")
    for map_id in new_maps:
        idx = product_atoms[map_id].GetIdx()
        for atom in result.GetAtoms():
            if atom.GetIdx() == idx:
                continue
            if (
                positions.GetAtomPosition(idx) - positions.GetAtomPosition(atom.GetIdx())
            ).Length() < 0.5:
                raise ValueError("An initialized added atom overlaps another heavy atom")
    return result


def initialize_product_pose(
    source_mapped_smiles: str,
    graph_result: GraphEditResult,
    source_pose_sdf: Path,
    output: Path,
    seed: int = 42,
) -> Path:
    """Write one mapped edited-graph pose and an explicit initialization receipt.

    For deletions/reconnections/substitutions there is no embedding or optimization.
    For additions, ETKDG and fixed-point relaxation are best-effort and may fail.
    In either case every surviving source heavy atom stays at its original position.
    """
    output, source_pose_sdf = output.resolve(), source_pose_sdf.resolve()
    receipt = output.with_suffix(".initialization.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Use a new product initialization output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "agenticgen.pose_initialization.v1",
        "status": "started",
        "seed": seed,
        "rdkit_version": rdBase.rdkitVersion,
        "source_pose_sdf": str(source_pose_sdf),
        "output_sdf": str(output),
        "source_mapped_smiles": source_mapped_smiles,
        "product_mapped_smiles": graph_result.product_mapped_smiles,
        "coordinate_frame": "source_pocket_frame",
        "is_flowr_output": False,
        "is_validated_bound_pose": False,
        "limitations": [
            "Coordinates are an initialization for native FLOWR inpainting, not a model prediction.",
            "Copying positions does not validate new bonds, stereochemistry or bound-pose strain.",
            "New atoms, when present, use a single seeded constrained ETKDG attempt and local relaxation.",
        ],
    }
    try:
        if not 0 <= seed <= 2147483647:
            raise ValueError("Initialization seed must be a nonnegative signed 32-bit integer")
        if graph_result.status == "rejected" or not graph_result.product_mapped_smiles:
            raise ValueError("Unparseable/unavailable SynCraft product is skipped, not repaired")
        source = molecule(source_mapped_smiles)
        product = molecule(graph_result.product_mapped_smiles)
        source_atoms, product_atoms = _atoms(source), _atoms(product)
        metadata["source_pose_sha256"] = sha256(source_pose_sdf)
        supplier = Chem.SDMolSupplier(str(source_pose_sdf), removeHs=True)
        if len(supplier) != 1 or supplier[0] is None:
            raise ValueError("Source pose SDF must contain exactly one readable molecule")
        source_pose = supplier[0]
        _check_mapped_graph(source, source_pose)
        if source_pose.GetNumConformers() != 1 or not source_pose.GetConformer().Is3D():
            raise ValueError("Source pose must contain a 3D conformer")
        source_positions = {
            a.GetAtomMapNum(): source_pose.GetConformer().GetAtomPosition(a.GetIdx())
            for a in source_pose.GetAtoms()
        }
        if not all(math.isfinite(v) for xyz in source_positions.values() for v in xyz):
            raise ValueError("Source pose coordinates must be finite")
        shared = set(source_atoms) & set(product_atoms)
        added = set(product_atoms) - set(source_atoms)
        deleted = set(source_atoms) - set(product_atoms)
        substituted = {
            i
            for i in shared
            if (source_atoms[i].GetAtomicNum(), source_atoms[i].GetFormalCharge())
            != (product_atoms[i].GetAtomicNum(), product_atoms[i].GetFormalCharge())
        }
        metadata.update(
            added_atom_maps=sorted(added),
            deleted_atom_maps=sorted(deleted),
            substituted_atom_maps=sorted(substituted),
            frozen_source_atom_maps=sorted(shared),
        )
        if added:
            metadata["method"] = "constrained_ETKDGv3_and_fixed_source_atom_relaxation"
            initialized = _initialize_new_atoms(
                product, product_atoms, source_positions, seed, metadata
            )
        else:
            metadata["method"] = "exact_surviving_source_coordinate_copy"
            initialized = _coordinate_copy(product, product_atoms, source_positions)
        positions = initialized.GetConformer()
        max_deviation = max(
            (
                (
                    positions.GetAtomPosition(product_atoms[i].GetIdx()) - source_positions[i]
                ).Length()
                for i in shared
            ),
            default=0.0,
        )
        if max_deviation > 1e-9:
            raise ValueError("Initialization moved a surviving source heavy atom")
        with Chem.SDWriter(str(output)) as writer:
            writer.write(initialized)
        reloaded = Chem.SDMolSupplier(str(output), removeHs=True)[0]
        if reloaded is None:
            raise ValueError("Initialized product SDF could not be read")
        _check_mapped_graph(product, reloaded)
        reloaded_atoms = _atoms(reloaded)
        serialized_deviation = max(
            (
                (
                    reloaded.GetConformer().GetAtomPosition(reloaded_atoms[i].GetIdx())
                    - source_positions[i]
                ).Length()
                for i in shared
            ),
            default=0.0,
        )
        if serialized_deviation > 1e-8:
            raise ValueError("Serialized initialization changed surviving source coordinates")
        metadata.update(
            status="completed",
            output_sha256=sha256(output),
            output_smiles=canonical_smiles(reloaded),
            max_surviving_source_displacement_angstrom=serialized_deviation,
        )
        write_json(receipt, metadata)
        return output
    except (ValueError, OSError, RuntimeError) as error:
        metadata.update(status="failed", reason=f"{type(error).__name__}: {error}")
        # An invalid initialization must never masquerade as a usable FLOWR input.
        if output.is_file():
            output.unlink()
        write_json(receipt, metadata)
        raise
