"""Native SynCraft designs -> native FLOWR inpainting -> external pose evaluation."""

from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem

from agenticgen.adapters.geometry import SynCraftGeometry
from agenticgen.adapters.properties import RDKitSAEvaluator
from agenticgen.base_config import FoundationConfiguration
from agenticgen.chemistry import canonical_smiles, graph_hash, similarity
from agenticgen.contracts import BindingResult, GraphEditResult, Observation
from agenticgen.factory import build_retrosynthesis
from agenticgen.inpainting import FLOWRInpaintingAdapter, plan_masks
from agenticgen.inpainting_contracts import InpaintingConfig, InpaintingRequest
from agenticgen.io import read_json, sha256, utc_now, write_json
from agenticgen.pose_quality import measure_pose
from agenticgen.selection import contact_retention


def _find_pose(source_run: Path, preferred: Path, mapped_smiles: str) -> Path:
    """A deduplicated foundation evaluation may store the pose in another case."""
    if preferred.is_file():
        return preferred
    expected = canonical_smiles(mapped_smiles, maps=True)
    for path in sorted(source_run.glob("cases/*/geometry_*/docked_pose.sdf")):
        supplier = Chem.SDMolSupplier(str(path), removeHs=True)
        if len(supplier) == 1 and supplier[0] is not None:
            if canonical_smiles(supplier[0], maps=True) == expected:
                return path
    return preferred


def load_inpainting(path: Path) -> InpaintingConfig:
    raw = read_json(path)
    for key in ("root", "python", "checkpoint"):
        expanded = os.path.expandvars(raw[key])
        if "${" in expanded or ("%" in expanded and expanded.count("%") >= 2):
            raise ValueError(f"Unresolved environment variable in {key}")
        value = Path(expanded).expanduser()
        raw[key] = str(value.resolve() if value.is_absolute() else (path.parent / value).resolve())
    return InpaintingConfig.model_validate(raw)


def inpainting_run(
    source_run: Path,
    config: InpaintingConfig,
    pocket_pdb: Path,
    output: Path,
    *,
    execute: bool = False,
    evaluation: FoundationConfiguration | None = None,
    redock: bool = False,
) -> dict:
    """Plan by default. Invalid edits never reach pose preparation or FLOWR.

    Evaluation consumes the generated SDF itself. Source and generated poses are
    scored with the same score-only protocol; optional independent redocking is
    saved separately and never substituted for the generated coordinates.
    """
    source_run, pocket_pdb, output = source_run.resolve(), pocket_pdb.resolve(), output.resolve()
    source_manifest = read_json(source_run / "manifest.json")
    if source_manifest.get("schema_version") not in {
        "agenticgen.foundation-run.v1",
        "agenticgen.design-run.v1",
    }:
        raise ValueError("Input must be a saved SynCraft foundation or design run")
    if evaluation is not None and not execute:
        raise ValueError("Evaluation requires --execute")
    if redock and evaluation is None:
        raise ValueError("Independent redocking requires an evaluation configuration")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "agenticgen.inpainting-run.v1",
        "status": "running",
        "started_at": utc_now(),
        "source_manifest_sha256": sha256(source_run / "manifest.json"),
        "source_execution_kind": source_manifest.get(
            "execution_kind", source_manifest.get("llm_backend")
        ),
        "execute": execute,
        "effective_config": config.model_dump(mode="json"),
        "evaluation_config": evaluation.value.model_dump(mode="json") if evaluation else None,
        "independent_redock": redock,
    }
    write_json(output / "manifest.json", manifest)
    adapter = FLOWRInpaintingAdapter(config)
    properties = RDKitSAEvaluator()
    geometry = SynCraftGeometry(evaluation) if evaluation else None
    retro = build_retrosynthesis(evaluation) if evaluation else None
    retro_cache = {}
    cases = []
    try:
        for result_path in sorted(source_run.glob("cases/*/result.json")):
            result = read_json(result_path)
            case_id, source_folder = result["case_id"], result_path.parent
            case_folder = output / "cases" / case_id
            observation = Observation.model_validate(read_json(source_folder / "observation.json"))
            source_binding = BindingResult.model_validate(result["source_binding"])
            interactions = tuple(
                sorted({i for c in source_binding.interactions for i in c.atom_maps})
            )
            source_pose = _find_pose(
                source_run,
                source_folder / "geometry_original/docked_pose.sdf",
                observation.mapped_smiles,
            )
            source_sa = properties.evaluate(observation.source_smiles)
            source_retro = None
            if retro:
                source_key = graph_hash(observation.source_smiles)
                if source_key not in retro_cache:
                    retro_cache[source_key] = retro.evaluate(
                        observation.source_smiles, case_folder / "retro_source"
                    )
                source_retro = retro_cache[source_key]
            source_scored = None
            if geometry and source_pose.is_file():
                source_scored = geometry.evaluate_pose(
                    observation.mapped_smiles, source_pose, case_folder / "source_score_only"
                )
            records = []
            for candidate in result["candidates"]:
                if "graph" not in candidate:
                    records.append(
                        {
                            "proposal_index": candidate["proposal_index"],
                            "status": "skipped",
                            "reason": "No reconstructed graph",
                        }
                    )
                    continue
                index = candidate["proposal_index"]
                folder = case_folder / f"proposal_{index:02d}"
                graph = GraphEditResult.model_validate(candidate["graph"])
                request = InpaintingRequest(
                    case_id=case_id,
                    proposal_id=f"proposal_{index:02d}",
                    source_mapped_smiles=observation.mapped_smiles,
                    graph_result=graph,
                    interaction_atom_maps=interactions,
                    pocket_pdb=str(pocket_pdb),
                )
                # A rejected/unreadable graph is an expected skip, not a repair input.
                if execute and any(p.status == "ready" for p in plan_masks(request, config)):
                    try:
                        pose = _find_pose(
                            source_run,
                            source_folder / f"geometry_product_{index:02d}/docked_pose.sdf",
                            graph.product_mapped_smiles,
                        )
                        if not pose.is_file():
                            from agenticgen.pose_initialization import initialize_product_pose

                            pose = initialize_product_pose(
                                observation.mapped_smiles,
                                graph,
                                source_pose,
                                folder / "initialized_product.sdf",
                                seed=config.seed,
                            )
                        request = request.model_copy(update={"product_pose_sdf": str(pose)})
                    except (OSError, ValueError, RuntimeError) as error:
                        record = {
                            "proposal_index": index,
                            "status": "failed",
                            "reason": f"Pose initialization: {error}",
                            "evaluations": [],
                        }
                        write_json(folder / "result.json", record)
                        records.append(record)
                        continue
                sampled = adapter.run(request, folder / "inpainting", execute=execute)
                evaluations = []
                for mode in sampled.mode_results:
                    for pose in mode.poses:
                        if not pose.accepted_for_evaluation:
                            evaluations.append(
                                {
                                    "pose_id": pose.pose_id,
                                    "status": "skipped",
                                    "reason": "Fixed-region preservation failed",
                                }
                            )
                            continue
                        pose_path = Path(pose.sdf_path)
                        try:
                            quality = measure_pose(pose_path)
                        except (OSError, ValueError, RuntimeError) as error:
                            quality = {"status": "error", "reason": str(error)}
                        measured = {
                            "pose_id": pose.pose_id,
                            "mode": pose.mode,
                            "pose_sdf": str(pose_path),
                            "status": "measured",
                            "mapped_smiles": pose.mapped_smiles,
                            "properties": properties.evaluate(pose.smiles).model_dump(mode="json"),
                            "geometry_quality": quality,
                            "source_similarity": similarity(
                                observation.source_smiles,
                                pose.smiles,
                                evaluation.value.selection.similarity_bits if evaluation else 2048,
                            ),
                        }
                        product_sa = measured["properties"]
                        measured["delta_sa_auxiliary"] = (
                            source_sa.value - product_sa["value"]
                            if source_sa.status == product_sa["status"] == "ok"
                            and source_sa.value is not None
                            and product_sa["value"] is not None
                            else None
                        )
                        if geometry:
                            scored = geometry.evaluate_pose(
                                pose.mapped_smiles, pose_path, folder / f"evaluation_{pose.pose_id}"
                            )
                            measured["binding_score_only"] = scored.model_dump(mode="json")
                            measured["contact_retention"] = contact_retention(
                                source_binding, scored
                            )
                            measured["contact_retention_score_only_baseline"] = contact_retention(
                                source_scored or source_binding, scored
                            )
                            measured["vina_loss_same_mode"] = (
                                scored.score_kcal_mol - source_scored.score_kcal_mol
                                if source_scored and source_scored.status == scored.status == "ok"
                                else None
                            )
                            key = graph_hash(pose.smiles)
                            if key not in retro_cache:
                                retro_cache[key] = retro.evaluate(
                                    pose.smiles, folder / f"retro_{pose.pose_id}"
                                )
                            measured["retrosynthesis"] = retro_cache[key].model_dump(mode="json")
                            if redock:
                                measured["independent_redock"] = geometry.evaluate_pose(
                                    pose.mapped_smiles,
                                    pose_path,
                                    folder / f"redock_{pose.pose_id}",
                                    mode="redock",
                                ).model_dump(mode="json")
                        evaluations.append(measured)
                record = {
                    "proposal_index": index,
                    "proposal_mode": candidate.get("proposal_mode"),
                    "status": sampled.status,
                    "inpainting_result": str(folder / "inpainting/result.json"),
                    "sampling_modes": [m.model_dump(mode="json") for m in sampled.mode_results],
                    "evaluations": evaluations,
                }
                write_json(folder / "result.json", record)
                records.append(record)
            saved = {
                "case_id": case_id,
                "source_sa": source_sa.model_dump(mode="json"),
                "source_retrosynthesis": source_retro.model_dump(mode="json")
                if source_retro
                else None,
                "source_score_only": source_scored.model_dump(mode="json")
                if source_scored
                else None,
                "proposals": records,
            }
            write_json(case_folder / "result.json", saved)
            cases.append(saved)
        if not cases:
            raise ValueError("No saved case results found")
        records = [r for c in cases for r in c["proposals"]]
        measured = [
            e for r in records for e in r.get("evaluations", []) if e["status"] == "measured"
        ]
        failed_eval = any(
            e.get("binding_score_only", {}).get("status") == "infrastructure_error"
            or e.get("retrosynthesis", {}).get("status") == "infrastructure_error"
            for e in measured
        ) or any(
            (c.get("source_score_only") or {}).get("status") == "infrastructure_error"
            or (c.get("source_retrosynthesis") or {}).get("status") == "infrastructure_error"
            for c in cases
        )
        summary = {
            "schema_version": "agenticgen.inpainting-summary.v1",
            "case_count": len(cases),
            "proposal_count": len(records),
            "skipped_proposal_count": sum(r["status"] == "skipped" for r in records),
            "failed_proposal_count": sum(r["status"] == "failed" for r in records),
            "measured_pose_count": len(measured),
            "delivered_pose_count": sum(
                m["delivered_samples"] for r in records for m in r.get("sampling_modes", [])
            ),
            "preservation_eligible_pose_count": sum(
                m["eligible_samples"] for r in records for m in r.get("sampling_modes", [])
            ),
            "evaluation_infrastructure_error": failed_eval,
            "binding_ok_count": sum(
                e.get("binding_score_only", {}).get("status") == "ok" for e in measured
            ),
            "verified_route_count": sum(
                e.get("retrosynthesis", {}).get("route_verified", False) for e in measured
            ),
            "optimization_winner": None,
            "interpretation": "Engineering execution; model sampling and evaluation are separate. A readable graph, mask pass, lower SA or Vina score alone does not prove synthesis or binding.",
        }
        write_json(output / "summary.json", summary)
        manifest["status"] = (
            "partial"
            if failed_eval or any(r["status"] in {"failed", "partial"} for r in records)
            else "completed"
        )
        return summary
    except BaseException as error:
        manifest.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        manifest["completed_at"] = utc_now()
        write_json(output / "manifest.json", manifest)
