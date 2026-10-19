"""Derive SynCraft foundation summaries from saved results without reevaluating molecules."""

from pathlib import Path

from agenticgen.chemistry import graph_hash
from agenticgen.io import read_json, write_json
from agenticgen.reporting import usage_from_receipts


def foundation_summary(results: list[dict], settings, output: Path) -> dict:
    records = [r for case in results for r in case["candidates"]]
    accepted = [r for r in records if r["status"] == "accepted"]
    return {
        "schema_version": "agenticgen.foundation-summary.v1",
        "case_count": len(results),
        "proposal_count": len(records),
        "accepted_edit_count": len(accepted),
        "graph_reconstruction": "syncraft_native",
        "fragment_pruned_proposal_count": sum(
            bool(r.get("graph", {}).get("was_fragmented")) for r in records
        ),
        "product_loading_failure_count": sum(
            r.get("graph", {}).get("error_stage") == "product_loading" for r in records
        ),
        "reconstruction_failure_count": sum(
            r.get("graph", {}).get("error_stage") == "upstream_reconstruction" for r in records
        ),
        "unique_accepted_product_count": len(
            {graph_hash(r["graph"]["product_smiles"]) for r in accepted}
        ),
        "proposal_modes": list(settings.proposal_modes),
        "pass_k_per_mode": settings.pass_k,
        "mode_proposal_counts": {
            mode: sum(r.get("proposal_mode", settings.mode) == mode for r in records)
            for mode in settings.proposal_modes
        },
        "route_validated_selected_case_count": sum(
            r.get("route_validated_selection", {}).get("selected_proposal_index") is not None
            for r in results
        ),
        "rejected_edit_count": sum(r["status"] == "rejected" for r in records),
        "unchanged_count": sum(r["status"] == "unchanged" for r in records),
        "interface_error_count": sum(r["status"] == "interface_error" for r in records),
        "sa_improvement_count_auxiliary": sum(r["sa_improved"] is True for r in accepted),
        "selected_case_count": sum(r["selected_product_smiles"] is not None for r in results),
        "source_already_route_found_count": sum(
            r["source_retrosynthesis"]["status"] == "route_found" for r in results
        ),
        "route_rescue_evaluable_case_count": sum(r["route_rescue"] is not None for r in results),
        "route_rescue_case_count": sum(r["route_rescue"] is True for r in results),
        "verified_route_rescue_evaluable_case_count": sum(
            r.get("verified_route_rescue") is not None for r in results
        ),
        "verified_route_rescue_case_count": sum(
            r.get("verified_route_rescue") is True for r in results
        ),
        "selection_policy": settings.selection.policy,
        "enhancements_enabled": settings.enhancements.enabled,
        "physical_retro_queries": sum(not r["cache_hits"]["retro"] for r in accepted)
        + sum(not r.get("source_cache_hits", {}).get("retro", False) for r in results)
        if settings.retro.backend != "disabled"
        else 0,
        "physical_geometry_queries": sum(not r["cache_hits"]["binding"] for r in accepted)
        + sum(not r.get("source_cache_hits", {}).get("binding", False) for r in results)
        if settings.geometry.backend != "disabled"
        else 0,
        "route_reported_found_count": sum(
            r["retrosynthesis"]["status"] == "route_found" for r in accepted
        ),
        "route_verified_count": sum(r["retrosynthesis"]["route_verified"] for r in accepted),
        "execution_kind": "engineering_replay" if settings.llm.backend == "replay" else "real_api",
        "full_method_runtime_evaluated": settings.llm.backend != "replay"
        and settings.selection.policy != "proposal_only"
        and all(r["status"] == "completed" for r in results),
        "usage": usage_from_receipts(output),
        "interpretation": "Selection success is profile-specific; SA alone does not certify a synthetic route",
    }


def rebuild_foundation_summary(output: Path) -> dict:
    from agenticgen.base_config import FoundationConfig

    manifest = read_json(output / "manifest.json")
    if manifest["schema_version"] != "agenticgen.foundation-run.v1":
        raise ValueError("This is not a SynCraft foundation run")
    raw = dict(manifest["effective_config"])
    recorded_editor = raw.pop("editor", "strict")
    settings = FoundationConfig.model_validate(raw)
    results = sorted(
        (read_json(p) for p in output.glob("cases/*/result.json")),
        key=lambda r: r["source"]["selection_rank"],
    )
    if not results:
        raise ValueError("No saved foundation case results")
    summary = foundation_summary(results, settings, output)
    summary["graph_reconstruction"] = manifest.get(
        "graph_reconstruction", "strict" if recorded_editor == "strict" else "syncraft_native"
    )
    write_json(output / "summary.json", summary)
    return summary
