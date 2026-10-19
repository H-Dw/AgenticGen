"""Bounded single-round orchestration with immutable decisions before outcome evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticgen.chemistry import graph_hash, mapped_smiles
from agenticgen.config import Configuration
from agenticgen.contracts import CaseDataset, Observation, SourceCase
from agenticgen.factory import Components
from agenticgen.io import object_hash, sha256, utc_now, write_json
from agenticgen.reporting import summarize_results, usage_from_receipts


def observation_for(
    case: SourceCase, config: Configuration, source_sa: float | None
) -> Observation:
    # Deliberate whitelist: no cohort labels, selection rank, routes or future outcomes.
    visible = {
        key: case.source_metrics[key]
        for key in ("pic50_on_rescore", "mmff_relief_per_heavy", "strain_MMFF94s_kcal_mol")
        if key in case.source_metrics
    }
    if source_sa is not None:
        visible["sa_score_current_rdkit"] = source_sa
    return Observation(
        case_id=case.case_id,
        target=case.target,
        source_smiles=case.smiles,
        mapped_smiles=mapped_smiles(case.smiles),
        graph_sha256=graph_hash(case.smiles),
        terminal_metrics=visible,
        constraints=config.value.constraints,
    )


def _case_run(
    case: SourceCase, config: Configuration, components: Components, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True)
    source_sa = components.properties.evaluate(case.smiles)
    observation = observation_for(case, config, source_sa.value)
    write_json(output / "observation.json", observation.model_dump(mode="json"))
    records: list[dict[str, Any]] = []
    proposals = []
    # All independent SynCraft proposals are committed before product properties / retro are seen.
    for index in range(config.value.proposals_per_case):
        try:
            proposal = components.agent.propose(
                observation, index, output / "calls" / f"proposal_{index:02d}.json"
            )
            proposals.append(proposal)
            records.append(
                {
                    "proposal_index": index,
                    "status": "proposed",
                    "proposal": proposal.model_dump(mode="json"),
                }
            )
        except Exception as error:
            proposals.append(None)
            records.append(
                {
                    "proposal_index": index,
                    "status": "interface_error",
                    "error_type": type(error).__name__,
                    "reason": "Agent call failed; inspect its request/receipt artifacts. No fallback was used.",
                }
            )
    write_json(
        output / "decision.json",
        {
            "created_at": utc_now(),
            "primary_proposal_index": 0,
            "selection_rule": "first_proposal_before_evaluation",
            "proposals": records,
        },
    )
    baseline_retro = components.retro.evaluate(case.smiles, output / "retro_original")
    for record, proposal in zip(records, proposals, strict=True):
        if proposal is None:
            continue
        edited = components.editor.apply(
            observation.mapped_smiles, proposal.edits, config.value.constraints
        )
        record.update(status=edited.status, graph=edited.model_dump(mode="json"))
        if edited.status != "accepted":
            record.update(sa_improved=False, constrained_route_success=False)
            continue
        properties = components.properties.evaluate(edited.product_smiles)
        retro = components.retro.evaluate(
            edited.product_smiles, output / f"retro_product_{record['proposal_index']:02d}"
        )
        delta_sa = (
            source_sa.value - properties.value
            if (
                source_sa.status == "ok"
                and properties.status == "ok"
                and source_sa.value is not None
                and properties.value is not None
            )
            else None
        )
        # An unverified search-reported route or missing backend cannot become primary success.
        route_success = (
            True
            if retro.status == "route_found" and retro.route_verified
            else False
            if retro.status == "no_route_within_budget"
            else None
        )
        record.update(
            properties=properties.model_dump(mode="json"),
            retrosynthesis=retro.model_dump(mode="json"),
            delta_sa=delta_sa,
            sa_improved=delta_sa > 0 if delta_sa is not None else None,
            constrained_route_success=route_success,
            product_geometry_status="not_evaluated",
            product_binding_status="not_evaluated",
        )
    eligible = [r for r in records if r["status"] == "accepted" and r.get("delta_sa") is not None]
    best_auxiliary = min(eligible, key=lambda r: r["properties"]["value"]) if eligible else None
    result = {
        "case_id": case.case_id,
        "status": "partial"
        if any(
            r["status"] == "interface_error"
            or r.get("retrosynthesis", {}).get("status") == "infrastructure_error"
            or r.get("properties", {}).get("status") == "error"
            for r in records
        )
        or baseline_retro.status == "infrastructure_error"
        or source_sa.status != "ok"
        else "completed",
        "source": case.model_dump(mode="json"),
        "source_sa": source_sa.model_dump(mode="json"),
        "source_retrosynthesis": baseline_retro.model_dump(mode="json"),
        "primary_proposal_index": 0,
        "candidates": records,
        "best_sa_proposal_index_auxiliary": best_auxiliary["proposal_index"]
        if best_auxiliary
        else None,
        "interpretation": "SA improvement is auxiliary; edited affinity/strain are not measured.",
    }
    write_json(output / "result.json", result)
    return result


def run(
    config: Configuration, dataset: CaseDataset, components: Components, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = config.value
    count = len(dataset.cases) * settings.proposals_per_case
    manifest: dict[str, Any] = {
        "schema_version": "agenticgen.run-manifest.v1",
        "started_at": utc_now(),
        "status": "running",
        "engineering_example": True,
        "llm_backend": settings.llm.backend,
        "effective_config": settings.model_dump(mode="json"),
        "graph_reconstruction": "syncraft_native",
        "edit_constraints_scope": "prompt_guidance_only",
        "dataset_sha256": sha256(config.resolve(settings.dataset)),
        "dependency_provenance": components.provenance,
        "budget": {
            "logical_proposals": count,
            "maximum_http_attempts": count
            * settings.llm.network_attempts
            * (1 + settings.llm.max_format_repairs)
            if settings.llm.backend != "replay"
            else 0,
            "maximum_retro_jobs": len(dataset.cases) + count
            if settings.retro.backend != "disabled"
            else 0,
        },
    }
    manifest["protocol_sha256"] = object_hash(
        {
            "config": manifest["effective_config"],
            "dependencies": components.provenance,
            "dataset": manifest["dataset_sha256"],
        }
    )
    write_json(output / "manifest.json", manifest)
    results = []
    try:
        for case in dataset.cases:
            result = _case_run(case, config, components, output / "cases" / case.case_id)
            results.append(result)
        summary = summarize_results(results, settings.llm.backend)
        summary["usage"] = usage_from_receipts(output)
        write_json(output / "summary.json", summary)
        products = [
            f"{r['graph']['product_smiles']}\t{result['case_id']}_p{r['proposal_index']}"
            for result in results
            for r in result["candidates"]
            if r["status"] == "accepted"
        ]
        (output / "products.smi").write_text(
            "\n".join(products) + ("\n" if products else ""), encoding="utf-8"
        )
        manifest["status"] = (
            "partial" if any(r["status"] != "completed" for r in results) else "completed"
        )
        return summary
    except BaseException as error:
        manifest.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        manifest["completed_at"] = utc_now()
        write_json(output / "manifest.json", manifest)
