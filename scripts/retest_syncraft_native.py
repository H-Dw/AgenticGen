"""Reexecute saved CK2 edit programs with native SynCraft; no new LLM calls."""

from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from agenticgen.adapters.loading import load_file
from agenticgen.adapters.properties import RDKitSAEvaluator
from agenticgen.chemistry import canonical_smiles
from agenticgen.contracts import EditConstraints
from agenticgen.editing import SynCraftEditor
from agenticgen.io import read_json, sha256, source_fingerprint, write_json

RUNS = (
    "ck2_live_20261015",
    "ck2_sa_validation_live",
    "syncraft_foundation_ck2_live_20261016",
    "ck2_official_dual_example",
    "ck2_official_release_example",
)


def retest(repo: Path, output: Path) -> dict:
    raw = read_json(repo / "examples/ck2_official/dual.json")
    root = (repo / "examples/ck2_official" / raw["syncraft"]["root"]).resolve()
    upstream = load_file(root / "src/utils.py", "native_retest")

    class Backend:
        @staticmethod
        def reconstruct(smiles, commands):
            return upstream.reconstruct_from_sequence(smiles, commands)

    editor, properties = SynCraftEditor(Backend()), RDKitSAEvaluator()
    output.mkdir(parents=True, exist_ok=False)
    results, source_hashes = [], {}
    for name in RUNS:
        for path in sorted((repo / "outputs" / name / "cases").glob("*/result.json")):
            old, observation = read_json(path), read_json(path.parent / "observation.json")
            source_hashes[path.as_posix()] = sha256(path)
            for proposal in old["candidates"]:
                commands = copy.deepcopy(proposal["proposal"]["edits"])
                graph = editor.apply(
                    observation["mapped_smiles"],
                    commands,
                    EditConstraints.model_validate(observation["constraints"]),
                )
                expected = upstream.reconstruct_from_sequence(
                    observation["mapped_smiles"], copy.deepcopy(commands)
                )
                assert graph.upstream_mapped_smiles == expected["smiles_with_maps"]
                assert graph.was_fragmented == expected["was_fragmented"]
                valid = Chem.MolFromSmiles(expected["smiles_with_maps"]) is not None
                assert valid == (graph.status != "rejected")
                if valid:
                    assert graph.product_smiles == canonical_smiles(expected["smiles_with_maps"])
                assert commands == proposal["proposal"]["edits"]
                score = properties.evaluate(graph.product_smiles) if valid else None
                results.append(
                    {
                        "source_run": name,
                        "case_id": old["case_id"],
                        "proposal_index": proposal["proposal_index"],
                        "proposal_mode": proposal.get("proposal_mode", "standard_or_local"),
                        "previous_status": proposal["status"],
                        "previous_reason": proposal.get("graph", {}).get("reason"),
                        "status": graph.status,
                        "source_sa": old["source_sa"]["value"],
                        "product_sa": score.value if score else None,
                        "was_fragmented": graph.was_fragmented,
                        "graph": graph.model_dump(mode="json"),
                        "properties": score.model_dump(mode="json") if score else None,
                        "edits": commands,
                        "old_result_path": path.as_posix(),
                    }
                )
    if not results:
        raise ValueError("No saved CK2 proposals were found")
    by_run = []
    for name in RUNS:
        rows = [r for r in results if r["source_run"] == name]
        by_run.append(
            {
                "run": name,
                "proposals": len(rows),
                "previous_accepted": sum(r["previous_status"] == "accepted" for r in rows),
                "native_accepted": sum(r["status"] == "accepted" for r in rows),
                "native_rejected": sum(r["status"] == "rejected" for r in rows),
                "unique_products": len(
                    {r["graph"]["product_smiles"] for r in rows if r["status"] == "accepted"}
                ),
                "accepted_fragment_pruning": sum(
                    r["status"] == "accepted" and r["was_fragmented"] for r in rows
                ),
            }
        )
    summary = {
        "schema_version": "agenticgen.native-retest.v1",
        "execution_kind": "saved_proposal_replay",
        "new_api_calls": 0,
        "new_docking_jobs": 0,
        "new_retrosynthesis_jobs": 0,
        "graph_reconstruction": editor.policy,
        "proposal_count": len(results),
        "previous_counts": dict(Counter(r["previous_status"] for r in results)),
        "native_counts": dict(Counter(r["status"] for r in results)),
        "upstream_parity_checked_count": len(results),
        "by_run": by_run,
        "source_hashes": source_hashes,
        "upstream": source_fingerprint(root, ["src/utils.py"]),
        "editor_sha256": sha256(repo / "src/agenticgen/editing.py"),
    }
    write_json(output / "results.json", results)
    write_json(output / "summary.json", summary)
    fields = [
        "source_run",
        "case_id",
        "proposal_index",
        "proposal_mode",
        "previous_status",
        "status",
        "source_sa",
        "product_sa",
        "was_fragmented",
        "previous_reason",
        "old_result_path",
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    assert all(sha256(Path(path)) == digest for path, digest in source_hashes.items())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with rdBase.BlockLogs():
        summary = retest(Path(__file__).resolve().parents[1], args.output.resolve())
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
