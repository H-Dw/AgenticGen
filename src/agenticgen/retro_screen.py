"""Evaluate a frozen target list through the existing retrosynthesis port, without an agent."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from agenticgen.chemistry import canonical_smiles, graph_hash
from agenticgen.io import sha256, utc_now, write_json
from agenticgen.ports import RetrosynthesisPort


def screen_targets(
    targets: Path, output: Path, evaluator: RetrosynthesisPort, protocol: dict[str, Any]
) -> dict[str, Any]:
    lines = targets.read_text(encoding="utf-8-sig").splitlines()
    # Blank lines or duplicate graphs would desynchronize the exported target index.
    if not lines or any(not line.strip() or len(line.split()) != 1 for line in lines):
        raise ValueError("Expected exactly one SMILES per line, without labels or blank lines")
    smiles = [canonical_smiles(line.strip()) for line in lines]
    if len(set(smiles)) != len(smiles):
        raise ValueError("Retrosynthesis target list contains duplicate molecular graphs")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "agenticgen.retro-screen.v1",
        "started_at": utc_now(),
        "status": "running",
        "targets_sha256": sha256(targets),
        "target_count": len(smiles),
        "protocol": protocol,
        "agent_calls": 0,
    }
    write_json(output / "manifest.json", manifest)
    results = []
    try:
        for index, value in enumerate(smiles):
            record = evaluator.evaluate(value, output / f"target_{index:04d}")
            results.append(
                {
                    "target_index_zero_based": index,
                    "smiles": value,
                    "graph_sha256": graph_hash(value),
                    "retrosynthesis": record.model_dump(mode="json"),
                }
            )
            write_json(output / "results.json", results)
        counts = Counter(row["retrosynthesis"]["status"] for row in results)
        summary = {
            "target_count": len(smiles),
            "status_counts": dict(counts),
            "verified_route_count": sum(row["retrosynthesis"]["route_verified"] for row in results),
            "interpretation": "Search failure within budget is not proof of synthetic impossibility",
            "agent_calls": 0,
        }
        write_json(output / "summary.json", summary)
        manifest["status"] = "partial" if counts["infrastructure_error"] else "completed"
        return summary
    except BaseException as error:
        manifest.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        manifest["completed_at"] = utc_now()
        write_json(output / "manifest.json", manifest)
