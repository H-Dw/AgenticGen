"""Plan masks and run the unmodified FLOWR graph/substructure generation entry point.

Unreadable SynCraft products are recorded and skipped, never repaired here. Planning
requires RDKit only. Sampling runs in the separately configured FLOWR interpreter.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from rdkit import Chem

from agenticgen.chemistry import canonical_smiles, molecule
from agenticgen.inpainting_contracts import (
    AtomSlot,
    InpaintingConfig,
    InpaintingModeResult,
    InpaintingRequest,
    InpaintingResult,
    MaskPlan,
    PoseArtifact,
)
from agenticgen.io import read_json, sha256, source_fingerprint, write_json


def _mapped_atoms(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    atoms = {a.GetAtomMapNum(): a for a in mol.GetAtoms()}
    if 0 in atoms or len(atoms) != mol.GetNumAtoms():
        raise ValueError("Every heavy atom needs a unique positive atom-map ID")
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError("Inpainting contracts use heavy atoms only")
    return atoms


def _identity(atom: Chem.Atom) -> tuple:
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
        atom.GetIsAromatic(),
    )


def _neighborhood(atom: Chem.Atom) -> tuple:
    return tuple(
        sorted(
            (b.GetOtherAtom(atom).GetAtomMapNum(), str(b.GetBondType())) for b in atom.GetBonds()
        )
    )


def _pose_graph_matches(graph: Chem.Mol, pose: Chem.Mol) -> bool:
    """Allow a pose to resolve unspecified stereochemistry, not contradict it."""
    return (
        graph.GetNumAtoms() == pose.GetNumAtoms()
        and graph.GetNumBonds() == pose.GetNumBonds()
        and pose.HasSubstructMatch(graph, useChirality=True)
    )


def _preserved_atom_identity(expected: Chem.Atom, actual: Chem.Atom) -> bool:
    def fields(atom):
        return (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetIsAromatic(),
        )

    if fields(expected) != fields(actual):
        return False
    if expected.HasProp("_CIPCode"):
        return actual.HasProp("_CIPCode") and expected.GetProp("_CIPCode") == actual.GetProp(
            "_CIPCode"
        )
    return True


def plan_masks(request: InpaintingRequest, config: InpaintingConfig) -> list[MaskPlan]:
    """Use actual native reconstruction, not requested commands or atom_map - 1.

    Fragment inference reorders fixed atoms to the first output slots. Regenerated
    slots have no inherited molecular identity, even if the atom count is constant.
    """
    try:
        result = request.graph_result
        if result.status == "rejected" or not result.product_mapped_smiles:
            raise ValueError(
                "Unparseable/unavailable SynCraft product is ignored; "
                + (result.reason or "no readable mapped product")
            )
        source = molecule(request.source_mapped_smiles)
        product = molecule(result.product_mapped_smiles)
        before, after = _mapped_atoms(source), _mapped_atoms(product)
    except (ValueError, RuntimeError, TypeError) as error:
        return [MaskPlan(mode=mode, status="skipped", reason=str(error)) for mode in config.modes]

    shared = before.keys() & after.keys()
    deleted = set(before) - set(after)
    changed = {
        i
        for i in shared
        if _identity(before[i]) != _identity(after[i])
        or _neighborhood(before[i]) != _neighborhood(after[i])
    }
    deletion_boundary = {
        neighbor.GetAtomMapNum()
        for i in deleted
        for neighbor in before[i].GetNeighbors()
        if neighbor.GetAtomMapNum() in after
    }
    declared = set(request.llm_optimized_atom_maps)
    unknown_declared = declared - (set(before) | set(after))
    optimized = (set(after) - set(before)) | changed | deletion_boundary | (declared & set(after))
    interaction = set(request.interaction_atom_maps)
    interaction_conflicts = {i for i in interaction if i not in shared or i in changed}
    unknown_interaction = interaction - set(before)
    deleted_interaction = interaction & deleted
    changed_interaction = interaction & changed
    plans = []
    for mode in config.modes:
        fixed = set(after) if mode == "graph" else optimized | interaction
        fixed &= set(after)
        fixed_indices = sorted(after[i].GetIdx() for i in fixed)
        variable_indices = sorted(a.GetIdx() for i, a in after.items() if i not in fixed)
        # Native fragment inference concatenates extracted fixed atoms and new noise.
        output_slots = {idx: slot for slot, idx in enumerate(fixed_indices)}
        skipped = None
        if mode == "fragment":
            if unknown_interaction:
                skipped = f"Unknown source interaction atom maps: {sorted(unknown_interaction)}"
            elif config.interaction_policy == "strict" and interaction_conflicts:
                skipped = (
                    "Interaction preservation conflicts with actual LLM graph edits at atom maps "
                    f"{sorted(interaction_conflicts)}; fragment generation is skipped"
                )
            elif unknown_declared:
                skipped = f"Unknown LLM optimization atom maps: {sorted(unknown_declared)}"
            elif not fixed_indices:
                skipped = "No fixed region; refusing native fallback to de novo generation"
            elif not variable_indices:
                skipped = "No regenerable region; refusing native fallback to de novo generation"
        warnings = []
        if deleted_interaction or changed_interaction:
            warnings.append(
                f"Interaction policy={config.interaction_policy}. Deleted original interaction "
                f"maps {sorted(deleted_interaction)} are released; changed maps "
                f"{sorted(changed_interaction)} are fixed in their edited chemical state. "
                "These changes are not evidence of original contact retention. New bioisostere "
                "contacts must be measured by the downstream interaction evaluator."
            )
        if deleted:
            warnings.append(
                "Deleted atoms cannot be fixed. Surviving deletion-boundary atoms are fixed; "
                "their attachment bond counts/types are checked after sampling. This does not "
                "prove that an isomorphic deleted motif cannot reappear elsewhere."
            )
        if mode == "graph" and interaction_conflicts:
            warnings.append(
                "Graph inpainting keeps the edited graph, including changes to original "
                f"interaction atoms {sorted(interaction_conflicts)}; it cannot restore them."
            )
        pair_bonds = []
        for bond in product.GetBonds():
            left, right = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
            if left in fixed and right in fixed:
                pair_bonds.append([min(left, right), max(left, right), str(bond.GetBondType())])
        attachments = {
            i: sorted(
                str(bond.GetBondType())
                for bond in after[i].GetBonds()
                if bond.GetOtherAtom(after[i]).GetAtomMapNum() not in fixed
            )
            for i in sorted(fixed)
        }
        plans.append(
            MaskPlan(
                mode=mode,
                status="skipped" if skipped else "ready",
                reason=skipped,
                warnings=warnings,
                product_mapped_smiles=result.product_mapped_smiles,
                interaction_atom_maps=sorted(interaction),
                interaction_policy=config.interaction_policy,
                interaction_surviving_atom_maps=sorted(interaction & set(after)),
                interaction_deleted_atom_maps=sorted(deleted_interaction),
                interaction_changed_atom_maps=sorted(changed_interaction),
                optimized_atom_maps=sorted(optimized),
                deleted_atom_maps=sorted(deleted),
                deletion_boundary_atom_maps=sorted(deletion_boundary),
                fixed_atom_maps=sorted(fixed),
                fixed_input_slots=fixed_indices,
                regenerate_input_slots=variable_indices,
                cli_substructure_indices=variable_indices if mode == "fragment" else [],
                fixed_pair_bonds=sorted(pair_bonds),
                fixed_attachment_bonds=attachments,
                atom_slots=[
                    AtomSlot(
                        atom_map=i,
                        source_rdkit_index=before[i].GetIdx() if i in before else None,
                        product_rdkit_index=a.GetIdx(),
                        input_slot=a.GetIdx(),
                        flowr_output_slot=(
                            a.GetIdx() if mode == "graph" else output_slots.get(a.GetIdx())
                        ),
                        fixed=i in fixed,
                    )
                    for i, a in sorted(after.items(), key=lambda item: item[1].GetIdx())
                ],
            )
        )
    return plans


def prepare_product_pose(request: InpaintingRequest, output: Path) -> Path:
    """Reorder a mapped product pose without moving any coordinates.

    An unmapped pose is accepted only if there is exactly one full graph mapping.
    Symmetry-ambiguous poses must be supplied with the SynCraft product atom maps.
    """
    if request.product_pose_sdf is None:
        raise ValueError("A product pose SDF in the receptor coordinate frame is required")
    if not request.graph_result.product_mapped_smiles:
        raise ValueError("No readable product")
    product = molecule(request.graph_result.product_mapped_smiles)
    _mapped_atoms(product)
    supplier = Chem.SDMolSupplier(str(Path(request.product_pose_sdf).resolve()), removeHs=True)
    if request.product_sdf_record_index >= len(supplier):
        raise ValueError("Product SDF record index is out of range")
    pose = supplier[request.product_sdf_record_index]
    if pose is None or pose.GetNumAtoms() != product.GetNumAtoms():
        raise ValueError("Product pose cannot be read or has a different heavy-atom count")
    if not _pose_graph_matches(product, pose):
        raise ValueError("Product pose graph differs from the actual SynCraft product")
    if pose.GetNumConformers() == 0 or not pose.GetConformer().Is3D():
        raise ValueError("A 3D product pose is required; no conformer is fabricated")
    if not all(math.isfinite(float(v)) for xyz in pose.GetConformer().GetPositions() for v in xyz):
        raise ValueError("Product pose contains non-finite coordinates")
    pose_map_ids = [a.GetAtomMapNum() for a in pose.GetAtoms()]
    if all(pose_map_ids):
        pose_atoms = _mapped_atoms(pose)
        if set(pose_atoms) != set(_mapped_atoms(product)):
            raise ValueError("Product pose atom-map IDs differ from the edited graph")
        order = [pose_atoms[a.GetAtomMapNum()].GetIdx() for a in product.GetAtoms()]
    elif any(pose_map_ids):
        raise ValueError("Partially mapped product poses are ambiguous")
    else:
        query, target = Chem.Mol(product), Chem.Mol(pose)
        for mol in (query, target):
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        matches = target.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=2)
        if len(matches) != 1:
            raise ValueError("Unmapped product pose has ambiguous atom correspondence; supply maps")
        order = list(matches[0])
    ordered_pose = Chem.RenumberAtoms(pose, order)
    for atom, reference in zip(ordered_pose.GetAtoms(), product.GetAtoms(), strict=True):
        atom.SetAtomMapNum(reference.GetAtomMapNum())
    # Check the chosen map correspondence, including specified stereochemistry.
    identity_order = tuple(range(product.GetNumAtoms()))
    matches = ordered_pose.GetSubstructMatches(
        product, uniquify=False, useChirality=True, maxMatches=100000
    )
    if identity_order not in matches:
        raise ValueError("Pose atom-map assignment is inconsistent with product connectivity")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(output)) as writer:
        writer.SetForceV3000(True)
        writer.write(ordered_pose)
    return output


def native_arguments(
    config: InpaintingConfig, request: InpaintingRequest, plan: MaskPlan, ligand: Path, output: Path
) -> list[str]:
    """Construct a fixed native argument vector. No shell or arbitrary user flags."""
    if plan.status != "ready":
        raise ValueError("Cannot run a skipped mask plan")
    if not request.pocket_pdb:
        raise ValueError("A local pocket PDB is required")
    args = [
        "--pdb_file",
        str(Path(request.pocket_pdb).resolve()),
        "--ligand_file",
        str(ligand.resolve()),
        "--arch",
        "pocket",
        "--pocket_type",
        "holo",
        "--ckpt_path",
        str(Path(config.checkpoint).resolve()),
        "--save_dir",
        str(output.resolve()),
        "--seed",
        str(config.seed),
        "--gpus",
        str(config.gpus),
        "--num_workers",
        str(config.num_workers),
        "--batch_cost",
        str(config.batch_cost),
        "--max_sample_iter",
        str(config.max_sample_iter),
        "--sample_n_molecules_per_target",
        str(config.samples_per_mode),
        "--integration_steps",
        str(config.integration_steps),
        "--pocket_cutoff",
        str(config.pocket_cutoff),
        "--pocket_noise",
        "fix",
        "--final_inpaint",
    ]
    if config.cut_pocket:
        args.append("--cut_pocket")
    if plan.mode == "graph":
        args.extend(["--graph_inpainting", config.graph_initialization])
    else:
        args.extend(["--substructure_inpainting", "--substructure"])
        args.extend(str(i) for i in plan.cli_substructure_indices)
    # Do not enable canonicalization, size changes, alignment, post-hoc optimization,
    # uniqueness filters or a second structural generation algorithm.
    return args


def preservation_report(
    plan: MaskPlan, reference: Chem.Mol, generated: Chem.Mol, tolerance: float
) -> dict[str, Any]:
    """Validate native output slot identities and fixed atoms; never alter a result."""
    ref_atoms = _mapped_atoms(reference)
    fixed = {slot.atom_map: slot.flowr_output_slot for slot in plan.atom_slots if slot.fixed}
    errors = []
    differences = []
    boundary_changes = []
    free_boundary_changes = []
    checked_optimized_boundaries = []
    warnings = []
    hard_boundaries = set(plan.optimized_atom_maps) | set(plan.deletion_boundary_atom_maps)
    if generated.GetNumConformers() == 0 or not generated.GetConformer().Is3D():
        errors.append("Generated molecule lacks a 3D conformer")
    if generated.GetNumConformers() and not all(
        math.isfinite(float(v)) for xyz in generated.GetConformer().GetPositions() for v in xyz
    ):
        errors.append("Generated coordinates are not finite")
    if plan.mode == "graph" and not _pose_graph_matches(reference, generated):
        errors.append("Native graph inpainting did not preserve the entire edited graph")
    max_deviation = 0.0
    out_fixed_slots = {slot for slot in fixed.values() if slot is not None}
    for map_id, slot in fixed.items():
        if slot is None or slot >= generated.GetNumAtoms():
            differences.append(f"Missing output slot for fixed atom map {map_id}")
            continue
        expected, actual = ref_atoms[map_id], generated.GetAtomWithIdx(slot)
        if not _preserved_atom_identity(expected, actual):
            differences.append(f"Fixed atom identity differs at map {map_id}, slot {slot}")
        if plan.mode == "fragment":
            attachments = sorted(
                str(b.GetBondType())
                for b in actual.GetBonds()
                if b.GetOtherAtomIdx(slot) not in out_fixed_slots
            )
            expected_attachments = plan.fixed_attachment_bonds[map_id]
            boundary = {
                "atom_map": map_id,
                "expected_bond_types": expected_attachments,
                "observed_bond_types": attachments,
                "matched": attachments == expected_attachments,
                "hard_constraint": map_id in hard_boundaries,
            }
            if map_id in hard_boundaries:
                checked_optimized_boundaries.append(boundary)
            if not boundary["matched"]:
                boundary_changes.append(boundary)
                if map_id in hard_boundaries:
                    differences.append(f"Optimized boundary attachment differs at map {map_id}")
                else:
                    free_boundary_changes.append(boundary)
                    warnings.append(
                        f"External attachment changed at interaction-only fixed atom map {map_id}; "
                        "native substructure inpainting allows connections to the regenerated "
                        "region to change. Original contact retention requires downstream evaluation."
                    )
            if generated.GetNumConformers() and reference.GetNumConformers():
                left = reference.GetConformer().GetAtomPosition(expected.GetIdx())
                right = generated.GetConformer().GetAtomPosition(slot)
                deviation = float((left - right).Length())
                max_deviation = max(max_deviation, deviation)
                if deviation > tolerance:
                    differences.append(f"Fixed coordinate moved at map {map_id}: {deviation:.4f} A")
    # Check both bonds and non-bonds within the fixed induced graph.
    ids = sorted(fixed)
    for offset, left_id in enumerate(ids):
        for right_id in ids[offset + 1 :]:
            left_slot, right_slot = fixed[left_id], fixed[right_id]
            if left_slot is None or right_slot is None:
                continue
            if max(left_slot, right_slot) >= generated.GetNumAtoms():
                continue
            expected = reference.GetBondBetweenAtoms(
                ref_atoms[left_id].GetIdx(), ref_atoms[right_id].GetIdx()
            )
            actual = generated.GetBondBetweenAtoms(left_slot, right_slot)
            if (str(expected.GetBondType()) if expected else None) != (
                str(actual.GetBondType()) if actual else None
            ):
                differences.append(f"Fixed pair connectivity differs: {left_id}, {right_id}")
    errors.extend(differences)
    return {
        "mode": plan.mode,
        "passed": not errors,
        "fixed_atom_count": len(fixed),
        "fixed_coordinates_required": plan.mode == "fragment",
        "max_fixed_coordinate_deviation_angstrom": max_deviation,
        "coordinate_tolerance_angstrom": tolerance,
        "deleted_atom_maps": plan.deleted_atom_maps,
        "deletion_boundary_atom_maps": plan.deletion_boundary_atom_maps,
        "deletion_intent_scope": "fixed boundary attachment counts and bond types",
        "boundary_constraint_scope": "actual optimized atoms and surviving deletion boundaries only",
        "checked_optimized_boundaries": checked_optimized_boundaries,
        "boundary_changes": boundary_changes,
        "free_boundary_changes": free_boundary_changes,
        "warnings": warnings,
        "errors": errors,
    }


def collect_outputs(
    plan: MaskPlan, reference_sdf: Path, raw_sdf: Path, output: Path, config: InpaintingConfig
) -> InpaintingModeResult:
    """Keep the native SDF and expose individual preserved poses to evaluators."""
    reference = Chem.SDMolSupplier(str(reference_sdf), removeHs=True)[0]
    if reference is None:
        raise ValueError("Prepared reference SDF could not be reloaded")
    result = dict(mode=plan.mode, seed=config.seed, requested_samples=config.samples_per_mode)
    if not raw_sdf.is_file():
        return InpaintingModeResult(
            status="failed", reason="Native FLOWR produced no expected samples SDF", **result
        )
    poses = []
    reports = []
    supplier = Chem.SDMolSupplier(str(raw_sdf), removeHs=True)
    delivered = len(supplier)
    output.mkdir(parents=True, exist_ok=True)
    max_map = max(a.GetAtomMapNum() for a in reference.GetAtoms())
    for index, generated in enumerate(supplier):
        report_path = output / f"pose_{index:04d}.preservation.json"
        if generated is None:
            report = {"passed": False, "errors": ["Native SDF record is unparseable"]}
            write_json(report_path, report)
            reports.append(str(report_path))
            continue
        report = preservation_report(
            plan, reference, generated, config.coordinate_tolerance_angstrom
        )
        write_json(report_path, report)
        reports.append(str(report_path))
        # Only fixed slots inherit identity in fragment generation. New atoms receive
        # a separate namespace; they are never equated to regenerated source atoms.
        for atom in generated.GetAtoms():
            atom.SetAtomMapNum(max_map + 1000 + atom.GetIdx())
        for slot in plan.atom_slots:
            if (
                slot.flowr_output_slot is not None
                and slot.flowr_output_slot < generated.GetNumAtoms()
            ):
                generated.GetAtomWithIdx(slot.flowr_output_slot).SetAtomMapNum(slot.atom_map)
        sdf_path = output / f"pose_{index:04d}.sdf"
        with Chem.SDWriter(str(sdf_path)) as writer:
            writer.SetForceV3000(True)
            writer.write(generated)
        poses.append(
            PoseArtifact(
                pose_id=f"{plan.mode}_{index:04d}",
                mode=plan.mode,
                sdf_path=str(sdf_path),
                smiles=canonical_smiles(generated),
                mapped_smiles=canonical_smiles(generated, maps=True),
                accepted_for_evaluation=bool(report["passed"]),
                preservation_report_path=str(report_path),
            )
        )
    return InpaintingModeResult(
        status="completed" if poses else "failed",
        delivered_samples=delivered,
        eligible_samples=sum(p.accepted_for_evaluation for p in poses),
        poses=poses,
        raw_sdf_path=str(raw_sdf),
        artifact_paths=[str(raw_sdf), *reports, *(p.sdf_path for p in poses)],
        reason=None if poses else "Native output contained no readable molecules",
        **result,
    )


class FLOWRInpaintingAdapter:
    """Subprocess bridge, compatible with a pre-existing CUDA or DTK interpreter."""

    def __init__(self, config: InpaintingConfig) -> None:
        self.config = config

    def run(
        self, request: InpaintingRequest, output: Path, *, execute: bool = False
    ) -> InpaintingResult:
        output = output.resolve()
        if (output / "result.json").exists():
            raise FileExistsError("Use a new inpainting output directory to preserve receipts")
        output.mkdir(parents=True, exist_ok=True)
        plans = plan_masks(request, self.config)
        write_json(output / "request.json", request.model_dump(mode="json"))
        write_json(output / "config.json", self.config.model_dump(mode="json"))
        write_json(output / "mask_plans.json", [p.model_dump(mode="json") for p in plans])
        results = []
        for plan in plans:
            common = dict(
                mode=plan.mode,
                seed=self.config.seed,
                requested_samples=self.config.samples_per_mode,
            )
            if plan.status == "skipped":
                results.append(InpaintingModeResult(status="skipped", reason=plan.reason, **common))
            elif not execute:
                results.append(InpaintingModeResult(status="planned", **common))
            else:
                try:
                    results.append(self._execute(request, plan, output / plan.mode))
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    results.append(
                        InpaintingModeResult(
                            status="failed",
                            reason=str(error),
                            artifact_paths=[
                                str(path)
                                for path in sorted((output / plan.mode).rglob("*"))
                                if path.is_file()
                            ],
                            **common,
                        )
                    )
        statuses = {r.status for r in results}
        if statuses == {"skipped"}:
            status = "skipped"
        elif not execute:
            status = "planned"
        elif statuses <= {"completed", "skipped"} and "completed" in statuses:
            status = "completed" if statuses == {"completed"} else "partial"
        elif "completed" in statuses:
            status = "partial"
        else:
            status = "failed"
        artifacts = [
            str(output / name) for name in ("request.json", "config.json", "mask_plans.json")
        ]
        result = InpaintingResult(
            case_id=request.case_id,
            proposal_id=request.proposal_id,
            status=status,
            plans=plans,
            mode_results=results,
            artifact_paths=artifacts + [a for r in results for a in r.artifact_paths],
            reason="All modes skipped; see per-mode reasons" if status == "skipped" else None,
        )
        write_json(output / "result.json", result.model_dump(mode="json"))
        return result

    def _execute(
        self, request: InpaintingRequest, plan: MaskPlan, output: Path
    ) -> InpaintingModeResult:
        config = self.config
        root = Path(config.root).resolve()
        files = [
            "flowr/gen/generate_from_pdb.py",
            "flowr/data/interpolate.py",
            "flowr/models/mol_builder.py",
        ]
        for name in files:
            if not (root / name).is_file():
                raise ValueError(f"FLOWR source is missing: {name}")
        fingerprint = source_fingerprint(root, files)
        if config.expected_revision and fingerprint["revision"] != config.expected_revision:
            raise ValueError("Configured FLOWR revision does not match its checkout")
        if not Path(config.checkpoint).is_file():
            raise ValueError("FLOWR checkpoint is missing")
        if not request.pocket_pdb or not Path(request.pocket_pdb).is_file():
            raise ValueError("Input pocket PDB is missing")
        python = shutil.which(config.python)
        if python is None:
            raise ValueError("Configured FLOWR Python interpreter does not exist")
        output.mkdir(parents=True, exist_ok=True)
        ligand = prepare_product_pose(request, output / "input_product.sdf")
        sampling = output / "native"
        sampling.mkdir()
        args = native_arguments(config, request, plan, ligand, sampling)
        task = {
            "schema_version": "agenticgen.flowr_worker.v1",
            "root": str(root),
            "native_arguments": args,
            "mode": plan.mode,
            "seed": config.seed,
            "input_ligand": str(ligand),
            "mask_plan": plan.model_dump(mode="json"),
            "runtime_receipt": str(output / "runtime.json"),
        }
        write_json(output / "task.json", task)
        write_json(
            output / "provenance.json",
            {
                "source": fingerprint,
                "checkpoint_sha256": sha256(Path(config.checkpoint)),
                "input_sdf_sha256": sha256(ligand),
                "pocket_sha256": sha256(Path(request.pocket_pdb)),
                "seed": config.seed,
                "requested_samples": config.samples_per_mode,
                "integration_steps": config.integration_steps,
                "max_sample_iter": config.max_sample_iter,
                "native_arguments": args,
                "invalid_llm_graph_handling": "skip_without_repair",
                "native_decoder_policy": "upstream defaults, without external repair",
            },
        )
        worker = Path(__file__).parent / "workers" / "flowr_inpainting.py"
        with (output / "native.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [python, str(worker), str(output / "task.json")],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_seconds,
                check=False,
            )
        if process.returncode:
            detail = ""
            if (output / "runtime.json").is_file():
                detail = ": " + str(read_json(output / "runtime.json").get("error", ""))
            raise RuntimeError(
                f"Native FLOWR exited {process.returncode}{detail}; see {output / 'native.log'}"
            )
        raw = sampling / f"samples_{Path(request.pocket_pdb).stem}.sdf"
        collected = collect_outputs(plan, ligand, raw, output / "poses", config)
        return collected.model_copy(
            update={
                "artifact_paths": [
                    *collected.artifact_paths,
                    *(
                        str(output / name)
                        for name in (
                            "task.json",
                            "runtime.json",
                            "provenance.json",
                            "native.log",
                            "input_product.sdf",
                        )
                    ),
                ]
            }
        )
