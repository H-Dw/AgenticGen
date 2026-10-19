"""Measure every readable campaign particle without a selection or repair gate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rdkit import Chem

from agenticgen.adapters.geometry import SynCraftGeometry
from agenticgen.adapters.properties import RDKitSAEvaluator
from agenticgen.chemistry import canonical_smiles, similarity
from agenticgen.contracts import BindingResult
from agenticgen.io import read_json, sha256, write_json
from agenticgen.pose_quality import measure_pose
from agenticgen.selection import contact_retention


def posebusters(path: Path, receptor: Path) -> dict:
    try:
        from posebusters import PoseBusters
    except ImportError:
        return {"status": "unavailable", "reason": "PoseBusters is not installed"}
    try:
        molecule = Chem.SDMolSupplier(str(path), removeHs=True)[0]
        result = PoseBusters(config="dock").bust([molecule], None, str(receptor))
        if len(result) != 1:
            raise ValueError("PoseBusters returned the wrong number of rows")
        checks = {str(k): bool(v) for k, v in result.iloc[0].items()}
        return {"status": "ok", "pass": all(checks.values()), "checks": checks}
    except Exception as error:
        return {"status": "error", "reason": f"{type(error).__name__}: {error}"}


def interaction_displacement(
    source_pose: Path,
    product_pose: Path,
    interaction_maps: list[int],
    generated_to_reference: dict[int, int] | None = None,
) -> dict:
    source = Chem.SDMolSupplier(str(source_pose), removeHs=True)[0]
    product = Chem.SDMolSupplier(str(product_pose), removeHs=True)[0]
    if source is None or product is None:
        return {"status": "unavailable"}
    before = {a.GetAtomMapNum(): a for a in source.GetAtoms()}
    after = {
        (generated_to_reference or {}).get(a.GetAtomMapNum(), a.GetAtomMapNum()): a
        for a in product.GetAtoms()
    }
    displacements, missing, identity_changed = [], [], []
    for atom_map in sorted(set(interaction_maps)):
        if atom_map not in after:
            missing.append(atom_map)
            continue
        first, second = before[atom_map], after[atom_map]
        if (first.GetAtomicNum(), first.GetFormalCharge()) != (
            second.GetAtomicNum(),
            second.GetFormalCharge(),
        ):
            identity_changed.append(atom_map)
        a = np.asarray(source.GetConformer().GetAtomPosition(first.GetIdx()))
        b = np.asarray(product.GetConformer().GetAtomPosition(second.GetIdx()))
        displacements.append(
            {"atom_map": atom_map, "distance_angstrom": float(np.linalg.norm(a - b))}
        )
    return {
        "status": "ok",
        "coordinate_alignment": "none; common receptor frame",
        "missing_original_interaction_maps": missing,
        "changed_original_interaction_identity_maps": identity_changed,
        "displacements": displacements,
        "rmsd_angstrom": math.sqrt(np.mean([d["distance_angstrom"] ** 2 for d in displacements]))
        if displacements
        else None,
        "maximum_angstrom": max((d["distance_angstrom"] for d in displacements), default=None),
    }


def lineage_contacts(
    source_pose: Path, product_pose: Path, particle: dict, binding: BindingResult
) -> tuple[BindingResult, dict, dict]:
    """Slot lineage supports comparison; element/charge changes are not retention."""
    source = Chem.SDMolSupplier(str(source_pose), removeHs=True)[0]
    product = Chem.SDMolSupplier(str(product_pose), removeHs=True)[0]
    before = {a.GetAtomMapNum(): a for a in source.GetAtoms()}
    after = {a.GetAtomMapNum(): a for a in product.GetAtoms()}
    lineage = particle.get("slot_lineage", [])
    mapping = {int(r["generated_atom_map"]): int(r["reference_atom_map"]) for r in lineage}
    if not lineage:
        mapping = {i: i for i in after if i in before}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Particle slot lineage is not one-to-one")
    allowed, changed = {}, []
    for generated, reference in mapping.items():
        if generated not in after:
            raise ValueError("Lineage names an atom absent from the actual generated pose")
        if reference not in before:
            continue
        a, b = before[reference], after[generated]
        if (a.GetAtomicNum(), a.GetFormalCharge()) == (b.GetAtomicNum(), b.GetFormalCharge()):
            allowed[generated] = reference
        else:
            changed.append(reference)
    namespace = max([*before, *after], default=0) + 10000
    contacts = []
    for interaction in binding.interactions:
        maps = [allowed.get(i, namespace + i) for i in interaction.atom_maps]
        contacts.append(interaction.model_copy(update={"atom_maps": maps}))
    measured = binding.model_copy(update={"interactions": contacts})
    return (
        measured,
        mapping,
        {
            "definition": "tensor slot lineage; original element and charge must also match",
            "lineage_is_a_hard_identity_constraint": False,
            "actual_identity_matching_original_maps": sorted(allowed.values()),
            "actual_identity_changed_original_maps": sorted(changed),
            "product_reference_local_graph_matches": [
                {
                    "reference_atom_map": row["reference_atom_map"],
                    "matches": row.get("local_graph_matches"),
                }
                for row in lineage
            ],
        },
    )


def evaluate_particles(
    geometry: SynCraftGeometry,
    source_binding: BindingResult,
    source_pose: Path,
    source_smiles: str,
    particles: list[dict],
    output: Path,
    *,
    affinity: list[dict] | None = None,
) -> list[dict]:
    """Scientific failures receive rows; only infrastructure errors need job recovery."""
    output.mkdir(parents=True, exist_ok=True)
    properties = RDKitSAEvaluator()
    original_sa = properties.evaluate(source_smiles)
    affinity_by_id = {r["pose_id"]: r for r in (affinity or [])}
    interaction_maps = sorted({m for i in source_binding.interactions for m in i.atom_maps})
    rows = []
    for index, particle in enumerate(particles):
        pose_id = particle.get("pose_id", f"particle_{index:04d}")
        row = {
            "pose_id": pose_id,
            "particle_index": particle.get("particle_index", index),
            "generation": particle,
            "affinity": affinity_by_id.get(pose_id),
            "status": "unavailable",
        }
        rows.append(row)
        if not particle.get("sdf_path") or not Path(particle["sdf_path"]).is_file():
            row["reason"] = "No readable decoded SDF; retained in attempted denominator"
            continue
        path = Path(particle["sdf_path"])
        mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
        if mol is None:
            row["reason"] = "SDF parsing failed"
            continue
        before_hash = sha256(path)
        smiles, maps = canonical_smiles(mol), canonical_smiles(mol, maps=True)
        folder = output / pose_id
        folder.mkdir(parents=True, exist_ok=True)
        row.update(
            status="measured",
            pose_sdf=str(path),
            input_pose_sha256=before_hash,
            smiles=smiles,
            mapped_smiles=maps,
            connected=len(Chem.GetMolFrags(mol)) == 1,
            properties=properties.evaluate(smiles).model_dump(mode="json"),
        )
        if row["connected"]:
            row["source_similarity"] = similarity(source_smiles, smiles)
            response_path = folder / "score_only/response.json"
            if response_path.is_file():
                binding = BindingResult.model_validate(read_json(response_path))
                if (
                    binding.input_pose_sha256 != before_hash
                    or binding.evaluation_mode != "score_only"
                ):
                    raise ValueError("Cached pose evaluation is for a different input/protocol")
            else:
                binding = geometry.evaluate_pose(
                    maps, path, folder / "score_only", mode="score_only"
                )
            row["binding"] = binding.model_dump(mode="json")
            if binding.status == "infrastructure_error":
                row["status"] = "infrastructure_error"
            aligned, lineage_map, lineage_receipt = lineage_contacts(
                source_pose, path, particle, binding
            )
            row["lineage"] = lineage_receipt
            row["contact_retention"] = contact_retention(source_binding, aligned)
            directional = {"HydrogenBond", "SaltBridge", "PiStacking", "PiCation", "HalogenBond"}
            first = {
                (i.kind, i.residue, tuple(sorted(i.atom_maps)))
                for i in source_binding.interactions
                if i.kind in directional
            }
            second = {
                (i.kind, i.residue, tuple(sorted(i.atom_maps)))
                for i in aligned.interactions
                if i.kind in directional
            }
            row["directional_exact_recall"] = len(first & second) / len(first) if first else None
            row["delta_vina_score_only"] = (
                binding.score_kcal_mol - source_binding.score_kcal_mol
                if binding.score_kcal_mol is not None and source_binding.score_kcal_mol is not None
                else None
            )
        row["interaction_displacement"] = interaction_displacement(
            source_pose,
            path,
            interaction_maps,
            {
                int(r["generated_atom_map"]): int(r["reference_atom_map"])
                for r in particle.get("slot_lineage", [])
            },
        )
        try:
            row["pose_quality"] = measure_pose(path)
        except Exception as error:
            row["pose_quality"] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        row["posebusters"] = posebusters(path, geometry.files["receptor_pdb"])
        score = row["properties"]
        row["delta_sa_improvement"] = (
            original_sa.value - score["value"]
            if (
                original_sa.status == score["status"] == "ok"
                and original_sa.value is not None
                and score["value"] is not None
            )
            else None
        )
        if sha256(path) != before_hash:
            raise RuntimeError("An evaluator changed the actual generated pose")
        write_json(folder / "evaluation.json", row)
    write_json(output / "evaluations.json", {"attempted": len(particles), "particles": rows})
    if any(row["status"] == "infrastructure_error" for row in rows):
        raise RuntimeError("Geometry infrastructure failed; per-particle receipts are preserved")
    return rows
