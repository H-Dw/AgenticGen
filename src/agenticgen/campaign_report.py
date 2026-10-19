"""Coverage-first campaign reports; duplicates and invalid attempts are retained."""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from agenticgen.campaign_contracts import campaign_arms
from agenticgen.io import read_json, sha256, utc_now, write_json


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["case_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _median(rows: list[dict], key: str):
    values = [r[key] for r in rows if isinstance(r.get(key), int | float)]
    return statistics.median(values) if values else None


def report_campaign(campaign) -> dict:
    output = campaign.root / "report"
    output.mkdir(parents=True, exist_ok=True)
    jobs = campaign.ledger.list()
    source_affinity = {}
    for job in jobs:
        if job["stage"] == "source_affinity" and job["status"] == "completed":
            source_affinity = {
                r["pose_id"]: r for r in read_json(Path(job["result_path"]))["poses"]
            }
    originals, designs = {}, {}
    for job in jobs:
        if job["status"] != "completed":
            continue
        if job["stage"] == "source_geometry":
            originals[job["case_id"]] = read_json(Path(job["result_path"]))
        if job["stage"] == "design":
            designs[job["case_id"]] = read_json(Path(job["result_path"]))
    source_rows, proposal_rows = [], []
    for case in campaign.dataset.cases:
        baseline, design = originals.get(case.case_id, {}), designs.get(case.case_id, {})
        head = source_affinity.get(case.case_id, {})
        source_rows.append(
            {
                "case_id": case.case_id,
                "selection_rank": case.selection_rank,
                "source_smiles": case.smiles,
                "historical_selection_pic50": case.source_metrics.get("pic50_on_rescore"),
                "common_protocol_pic50": head.get("pic50"),
                "common_protocol_vina_score_only": baseline.get("binding", {}).get(
                    "score_kcal_mol"
                ),
                "sa": baseline.get("properties", {}).get("value"),
                "baseline_status": "measured" if baseline else "pending",
                "original_contact_count": len(baseline["binding"]["interactions"])
                if baseline
                else None,
                "design_status": design.get("status", "pending"),
                "chemical_proposal_count": design.get("chemical_proposal_count", 0),
                "retries": design.get("retry_count", 0),
                "edited_smiles": design.get("graph", {}).get("product_smiles"),
                "edited_sa": design.get("properties", {}).get("value"),
            }
        )
        for attempt in design.get("attempts", []):
            proposal = attempt.get("proposal", {})
            proposal_rows.append(
                {
                    "case_id": case.case_id,
                    "attempt": attempt["attempt_index"],
                    "status": attempt["status"],
                    "rationale": proposal.get("rationale"),
                    "reason": attempt.get("reason") or attempt.get("graph", {}).get("reason"),
                    "error_stage": attempt.get("graph", {}).get("error_stage"),
                    "response_path": attempt.get("response_path"),
                    "product_smiles": attempt.get("graph", {}).get("product_smiles"),
                }
            )
    evaluations = {}
    recovered_heads = {}
    generation_by_arm = defaultdict(list)
    for job in jobs:
        if job["stage"] == "generation":
            generation_by_arm[(job["case_id"], job["payload"]["arm_id"])].append(job)
        if job["stage"] == "affinity" and job["status"] == "completed":
            recovered_heads[
                (job["payload"]["generation_task_id"], job["payload"]["generation_result_sha256"])
            ] = read_json(Path(job["result_path"]))
        if job["stage"] == "evaluation" and job["status"] == "completed":
            evaluations[
                (job["payload"]["generation_task_id"], job["payload"]["generation_result_sha256"])
            ] = read_json(Path(job["result_path"]))
    particles, groups = [], defaultdict(list)
    for job in jobs:
        if job["stage"] != "generation" or job["status"] != "completed":
            continue
        result = read_json(Path(job["result_path"]))
        folder = Path(result["output_dir"])
        decoded = read_json(folder / "poses.json")["particles"]
        evaluated = evaluations.get((job["task_id"], sha256(Path(job["result_path"]))), {})
        measurements = {r["pose_id"]: r for r in evaluated.get("particles", [])}
        native_head = folder / "affinity.json"
        current_head = recovered_heads.get((job["task_id"], sha256(Path(job["result_path"]))))
        if current_head is None:
            current_head = read_json(native_head) if native_head.exists() else {}
        heads = {r["pose_id"]: r for r in current_head.get("poses", [])}
        for particle in decoded:
            metric = measurements.get(particle["pose_id"], {})
            head = heads.get(particle["pose_id"]) or metric.get("affinity") or {}
            before = source_affinity.get(job["case_id"], {}).get("pic50")
            pic50 = head.get("pic50")
            retention = metric.get("contact_retention", {})
            displacement = metric.get("interaction_displacement", {})
            row = {
                "case_id": job["case_id"],
                "arm_id": job["payload"]["arm_id"],
                "batch_index": job["payload"]["batch_index"],
                "seed": job["payload"]["seed"],
                "pose_id": particle["pose_id"],
                "particle_index": particle["particle_index"],
                "decode_status": particle["status"],
                "evaluation_status": metric.get("status", "pending"),
                "smiles": particle.get("smiles"),
                "pose_sdf": particle.get("sdf_path"),
                "pic50_status": head.get("status", "pending"),
                "pic50": pic50,
                "delta_pic50": pic50 - before if pic50 is not None and before is not None else None,
                "vina_score_only": metric.get("binding", {}).get("score_kcal_mol"),
                "delta_vina_score_only": metric.get("delta_vina_score_only"),
                "sa": metric.get("properties", {}).get("value"),
                "sa_improvement": metric.get("delta_sa_improvement"),
                "exact_contact_recall": retention.get("fraction"),
                "paper_contact_recall": retention.get("paper_recall"),
                "directional_exact_recall": metric.get("directional_exact_recall"),
                "interaction_rmsd_angstrom": displacement.get("rmsd_angstrom"),
                "missing_original_interaction_atoms": len(
                    displacement.get("missing_original_interaction_maps", [])
                )
                if displacement
                else None,
                "identity_changed_original_interaction_atoms": len(
                    displacement.get("changed_original_interaction_identity_maps", [])
                )
                if displacement
                else None,
                "posebusters_pass": metric.get("posebusters", {}).get("pass"),
                "mmff_relief_kcal_mol": metric.get("pose_quality", {})
                .get("energy", {})
                .get("relief_kcal_mol"),
            }
            particles.append(row)
            groups[(job["case_id"], job["payload"]["arm_id"])].append(row)
    arm_rows = []
    for case in campaign.dataset.cases:
        if designs.get(case.case_id, {}).get("status") != "accepted":
            continue
        for arm in campaign_arms(campaign.config):
            rows = groups[(case.case_id, arm.arm_id)]
            arm_jobs = generation_by_arm[(case.case_id, arm.arm_id)]
            inapplicable = [
                j
                for j in arm_jobs
                if j["status"] == "skipped" or j["payload"].get("not_applicable")
            ]
            not_applicable = bool(arm_jobs) and len(inapplicable) == len(arm_jobs)
            good = [r for r in rows if r["decode_status"] == "decoded"]
            smiles = [r["smiles"] for r in good if r["smiles"]]
            arm_rows.append(
                {
                    "case_id": case.case_id,
                    "arm_id": arm.arm_id,
                    "nominal_attempts": 100,
                    "planned_attempts": 0 if not_applicable else 100,
                    "applicability": "not_applicable" if not_applicable else "applicable",
                    "not_applicable_reason": inapplicable[0]["payload"].get("condition_reason")
                    if not_applicable
                    else None,
                    "observed_attempts": len(rows),
                    "decoded": len(good),
                    "decode_fraction_of_observed_attempts": len(good) / len(rows) if rows else None,
                    "unique_graphs": len(set(smiles)),
                    "duplicates": len(smiles) - len(set(smiles)),
                    "affinity_scored": sum(r["pic50_status"] == "ok" for r in rows),
                    "vina_scored": sum(r["vina_score_only"] is not None for r in rows),
                    "median_pic50": _median(rows, "pic50"),
                    "median_delta_pic50": _median(rows, "delta_pic50"),
                    "median_vina_score_only": _median(rows, "vina_score_only"),
                    "median_delta_vina_score_only": _median(rows, "delta_vina_score_only"),
                    "median_sa_improvement": _median(rows, "sa_improvement"),
                    "median_exact_contact_recall": _median(rows, "exact_contact_recall"),
                    "median_interaction_rmsd_angstrom": _median(rows, "interaction_rmsd_angstrom"),
                }
            )
    summary = {
        "schema_version": "agenticgen.campaign-report.v1",
        "updated_at": utc_now(),
        "jobs": campaign.ledger.counts(),
        "frozen_cases": len(campaign.cases),
        "design_outcomes": dict(Counter(r.get("status") for r in designs.values())),
        "observed_generation_attempts": len(particles),
        "decoded_particles": sum(r["decode_status"] == "decoded" for r in particles),
        "maximum_generation_attempts": 32000,
        "denominator": "100 attempted slots per applicable arm; no replacement or deduplication",
        "interpretation": "Partial runs show observed coverage separately; pending work is not a scientific failure. Predicted pIC50, Vina score_only and SA are computational proxies.",
    }
    write_json(output / "summary.json", summary)
    _csv(output / "sources.csv", source_rows)
    _csv(output / "proposals.csv", proposal_rows)
    _csv(output / "particles.csv", particles)
    _csv(output / "arms.csv", arm_rows)
    text = [
        "# CK2 original-pose campaign",
        "",
        f"Frozen cases: {len(campaign.cases)}. Observed generation attempts: {len(particles)}.",
        "",
        "The original Steer coordinates define interaction conditions. All final poses use the same fixed-seed singleton self-conditioning affinity protocol and Vina score_only without redocking.",
        "",
        "The historical ranking pIC50 is retained as provenance and is not substituted for the newly scored baseline. No SA/Vina/validity gate removes attempted particles from these tables.",
        "",
        "See sources.csv, proposals.csv, particles.csv and arms.csv. A partial table is coverage evidence, not a completed comparative result.",
    ]
    (output / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return {**summary, "report_dir": str(output)}
