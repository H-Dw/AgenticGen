"""Rebuild descriptive CK2 SA tables from immutable saved decisions, without API calls."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from agenticgen.chemistry import canonical_smiles
from agenticgen.cohorts import write_csv
from agenticgen.io import read_json, sha256, write_json


def summarize(output: Path) -> dict:
    paths = sorted(output.glob("cases/*/result.json"))
    results = sorted((read_json(p) for p in paths), key=lambda r: r["source"]["selection_rank"])
    if not results:
        raise ValueError("No saved case results")
    rows, products = [], {}
    for result in results:
        source = result["source"]
        candidate = result["candidates"][result["primary_proposal_index"]]
        graph = candidate.get("graph", {})
        row = {
            "case_id": result["case_id"],
            "stratum": source["provenance"]["cohort"],
            "source_smiles": source["smiles"],
            **source["source_metrics"],
            "source_sa_recomputed": result["source_sa"]["value"],
            "edit_status": candidate["status"],
            "product_smiles": graph.get("product_smiles"),
            "product_sa": candidate.get("properties", {}).get("value"),
            "delta_sa_source_minus_product": candidate.get("delta_sa"),
            "similarity": graph.get("similarity"),
            "rejection_reason": graph.get("reason"),
            "product_binding_status": candidate.get("product_binding_status", "not_evaluated"),
            "product_geometry_status": candidate.get("product_geometry_status", "not_evaluated"),
            "source_route_status": result["source_retrosynthesis"]["status"],
            "product_route_status": candidate.get("retrosynthesis", {}).get("status", "not_run"),
        }
        rows.append(row)
        if graph.get("product_smiles"):
            key = canonical_smiles(graph["product_smiles"])
            products.setdefault(key, []).append(result["case_id"])
    write_csv(output / "comparison.csv", rows)
    summaries = {}
    for stratum in dict.fromkeys(row["stratum"] for row in rows):
        group = [r for r in rows if r["stratum"] == stratum]
        deltas = [
            r["delta_sa_source_minus_product"]
            for r in group
            if r["delta_sa_source_minus_product"] is not None
        ]
        failures = sum(r["edit_status"] in {"rejected", "unchanged"} for r in group)
        observed = deltas + [0.0] * failures
        summaries[stratum] = {
            "n_assigned": len(group),
            "n_accepted_with_sa": len(deltas),
            "n_rejected_or_unchanged": failures,
            "n_sa_improved": sum(value > 0 for value in deltas),
            "mean_delta_sa_accepted": statistics.mean(deltas) if deltas else None,
            "mean_delta_sa_with_failed_edits_zero": statistics.mean(observed) if observed else None,
            "n_observed_for_failure_zero_mean": len(observed),
        }
    write_json(
        output / "strata_summary.json",
        {
            "strata": summaries,
            "meaning": "Descriptive pilot; positive delta is improvement; negative deltas retained. "
            "Rejected/unchanged edits count zero; infrastructure errors remain missing.",
            "result_hashes": {p.relative_to(output).as_posix(): sha256(p) for p in paths},
        },
    )
    (output / "retro_products.smi").write_text(
        "".join(s + "\n" for s in products), encoding="utf-8"
    )
    if products:
        write_csv(
            output / "retro_products_index.csv",
            [
                {
                    "target_index_zero_based": i,
                    "source_case_ids": ";".join(ids),
                    "smiles": smiles,
                    "status": "not_run",
                }
                for i, (smiles, ids) in enumerate(products.items())
            ],
        )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    print(summarize(parser.parse_args().run.resolve()))
