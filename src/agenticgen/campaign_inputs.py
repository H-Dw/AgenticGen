"""Original-pose correspondence and graph-only placeholders, without geometry gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from agenticgen.chemistry import mapped_smiles, molecule
from agenticgen.contracts import GraphEditResult
from agenticgen.io import sha256, write_json


def write_original_pose(raw: Chem.Mol, source_smiles: str, output: Path) -> dict:
    """Assign authoritative maps by a full isomorphism; retain raw heavy coordinates."""
    source = molecule(mapped_smiles(source_smiles))
    pose = Chem.RemoveHs(Chem.Mol(raw))
    matches = pose.GetSubstructMatches(source, useChirality=True, uniquify=False, maxMatches=100001)
    if not matches or len(matches) > 100000 or pose.GetNumAtoms() != source.GetNumAtoms():
        raise ValueError("No complete enumerable source-pose correspondence")
    match = min(matches)
    for atom in pose.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom, index in zip(source.GetAtoms(), match, strict=True):
        pose.GetAtomWithIdx(index).SetAtomMapNum(atom.GetAtomMapNum())
    # Put source and graph slots in the same deterministic order without fitting coordinates.
    pose = Chem.RenumberAtoms(pose, list(match))
    output.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(output)) as writer:
        writer.SetForceV3000(True)
        writer.write(pose)
    check = Chem.SDMolSupplier(str(output), removeHs=True)[0]
    if check is None or not np.allclose(
        check.GetConformer().GetPositions(), pose.GetConformer().GetPositions(), atol=1e-4
    ):
        raise ValueError("Source pose serialization changed coordinates")
    receipt = {
        "schema_version": "agenticgen.original-pose.v1",
        "isomorphism_count": len(matches),
        "selected_source_to_raw_indices": list(match),
        "coordinate_alignment": "none",
        "mapped_smiles": mapped_smiles(source_smiles),
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".mapping.json"), receipt)
    return receipt


def write_graph_placeholder(
    graph: GraphEditResult, source_pose: Path, output: Path, seed: int
) -> Path:
    """Only I may later constrain positions; all other values are sampler placeholders."""
    if graph.status != "accepted" or not graph.product_mapped_smiles:
        raise ValueError("A parsable changed product graph is required")
    product = molecule(graph.product_mapped_smiles)
    source = Chem.SDMolSupplier(str(source_pose), removeHs=True)[0]
    if source is None:
        raise ValueError("Original pose unavailable")
    original = {
        a.GetAtomMapNum(): np.asarray(source.GetConformer().GetAtomPosition(a.GetIdx()))
        for a in source.GetAtoms()
    }
    center = np.asarray(list(original.values())).mean(axis=0)
    rng = np.random.default_rng(seed)
    conformer = Chem.Conformer(product.GetNumAtoms())
    conformer.Set3D(True)
    for atom in product.GetAtoms():
        xyz = original.get(atom.GetAtomMapNum())
        if xyz is None:
            xyz = center + rng.normal(size=3) * 2
        conformer.SetAtomPosition(atom.GetIdx(), xyz)
    product.RemoveAllConformers()
    product.AddConformer(conformer)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(output)) as writer:
        writer.SetForceV3000(True)
        writer.write(product)
    write_json(
        output.with_suffix(".initialization.json"),
        {
            "schema_version": "agenticgen.graph-placeholder.v1",
            "seed": seed,
            "source_pose_sha256": sha256(source_pose),
            "is_evaluation_pose": False,
            "method": "survivor-copy and Gaussian new-atom placeholders; free priors overwritten",
            "geometry_embedding_required": False,
        },
    )
    return output
