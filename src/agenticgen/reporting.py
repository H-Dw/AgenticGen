"""Recomputable summaries from saved outcomes; no agent or molecular rerun."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agenticgen.io import object_hash, read_json, sha256, utc_now, write_json


def summarize_results(results: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    candidates = [row for result in results for row in result["candidates"]]
    primary = [result["candidates"][0] for result in results]
    return {
        "summary_schema": "agenticgen.summary.v2",
        "engineering_example": True,
        "llm_backend": backend,
        "case_count": len(results),
        "proposal_count": len(candidates),
        "accepted_edit_count": sum(r["status"] == "accepted" for r in candidates),
        "graph_reconstruction": "syncraft_native",
        "fragment_pruned_proposal_count": sum(
            bool(r.get("graph", {}).get("was_fragmented")) for r in candidates
        ),
        "product_loading_failure_count": sum(
            r.get("graph", {}).get("error_stage") == "product_loading" for r in candidates
        ),
        "rejected_edit_count": sum(r["status"] == "rejected" for r in candidates),
        "interface_error_count": sum(r["status"] == "interface_error" for r in candidates),
        "sa_improvement_count_auxiliary": sum(r.get("sa_improved") is True for r in candidates),
        "primary_route_success_count": sum(
            r.get("constrained_route_success") is True for r in primary
        ),
        "primary_route_evaluated_count": sum(
            r.get("retrosynthesis", {}).get("status") in {"route_found", "no_route_within_budget"}
            for r in primary
        ),
        "primary_constraint_failure_count": sum(
            r["status"] in {"rejected", "unchanged"} for r in primary
        ),
        "primary_outcome_available_count": sum(
            r.get("constrained_route_success") is not None for r in primary
        ),
        "binding_and_geometry_evaluated": False,
        "cases": [
            {
                "case_id": r["case_id"],
                "status": r["status"],
                "result": f"cases/{r['case_id']}/result.json",
            }
            for r in results
        ],
    }


def usage_from_receipts(output: Path) -> dict[str, Any]:
    total = {
        "successful_api_calls": 0,
        "http_attempts_in_success_receipts": 0,
        "reported_prompt_tokens": 0,
        "reported_completion_tokens": 0,
    }
    for path in output.glob("cases/*/calls/*.receipt.json"):
        receipt = read_json(path)
        if (
            re.fullmatch(r"proposal_\d+\.receipt\.json", path.name)
            and receipt.get("backend") == "real_api"
        ):
            total["successful_api_calls"] += 1
            total["http_attempts_in_success_receipts"] += receipt.get("attempts", 0)
        usage = receipt.get("usage") or {}
        total["reported_prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["reported_completion_tokens"] += usage.get("completion_tokens", 0)
    return {
        **total,
        "unreported_usage": "Not inferred for requests without provider usage receipts",
    }


def rebuild_summary(output: Path) -> dict[str, Any]:
    manifest = read_json(output / "manifest.json")
    paths = sorted(output.glob("cases/*/result.json"))
    results = sorted(
        (read_json(path) for path in paths), key=lambda r: r["source"]["selection_rank"]
    )
    if not results:
        raise ValueError("No completed case results to summarize")
    summary = summarize_results(results, manifest["llm_backend"])
    summary["graph_reconstruction"] = manifest.get("graph_reconstruction", "strict")
    summary["usage"] = usage_from_receipts(output)
    previous = read_json(output / "summary.json") if (output / "summary.json").exists() else None
    if previous != summary:
        write_json(
            output / "summary_rebuild.json",
            {
                "created_at": utc_now(),
                "previous_summary_content_sha256": object_hash(previous),
                "new_summary_content_sha256": object_hash(summary),
                "result_files": {str(p.relative_to(output)): sha256(p) for p in paths},
                "reason": "Separate constraint-failure scoring from actual route evaluation counts",
                "new_api_calls": 0,
            },
        )
        write_json(output / "summary.json", summary)
    return summary
