"""Apply the existing SynCraft selection policies to evaluated native poses."""

from pathlib import Path

from agenticgen.base_config import SelectionConfig
from agenticgen.contracts import BindingResult
from agenticgen.io import read_json, sha256, write_json
from agenticgen.selection import select_candidates


def select_poses(case: dict, settings: SelectionConfig) -> dict:
    baseline = case.get("source_score_only")
    if not baseline:
        return {
            "selected_pose_id": None,
            "route_validated_pose_id": None,
            "reason": "Source pose baseline is unavailable",
        }
    rows, labels = [], {}
    for proposal in case["proposals"]:
        for pose in proposal.get("evaluations", []):
            if pose["status"] != "measured":
                continue
            index = len(rows)
            labels[index] = f"proposal_{proposal['proposal_index']:02d}/{pose['pose_id']}"
            rows.append(
                {
                    "proposal_index": index,
                    "status": "accepted",
                    "selection_similarity": pose["source_similarity"],
                    "graph": {"product_smiles": pose["mapped_smiles"]},
                    "properties": pose["properties"],
                    "binding": pose.get("binding_score_only", {"status": "not_run"}),
                    "retrosynthesis": pose.get("retrosynthesis", {"status": "not_run"}),
                }
            )
    original = BindingResult.model_validate(baseline)
    primary = select_candidates(rows, settings, original)
    route = select_candidates(
        [
            r
            for r in rows
            if r["retrosynthesis"].get("status") == "route_found"
            and r["retrosynthesis"].get("route_verified")
        ],
        settings,
        original,
    )
    return {
        "selected_pose_id": labels.get(primary["selected_proposal_index"]),
        "route_validated_pose_id": labels.get(route["selected_proposal_index"]),
        "pose_index": labels,
        "primary": primary,
        "route_validated": route,
        "binding_comparison": "source and generated coordinates, both Vina score_only",
        "geometry_quality_scope": "reported separately; native SynCraft policy has no MMFF gate",
    }


def rebuild_inpainting_selection(output: Path) -> dict:
    """Rebuild derived selection only; never repeat calls or alter raw results."""
    manifest = read_json(output / "manifest.json")
    if manifest["schema_version"] != "agenticgen.inpainting-run.v1":
        raise ValueError("Expected a saved inpainting run")
    evaluation = manifest.get("evaluation_config")
    settings = (
        SelectionConfig.model_validate(evaluation["selection"])
        if evaluation
        else SelectionConfig(policy="proposal_only")
    )
    selected = {}
    for path in sorted(output.glob("cases/*/result.json")):
        case = read_json(path)
        decision = select_poses(case, settings)
        write_json(path.parent / "selection.json", decision)
        if decision["selected_pose_id"]:
            selected[case["case_id"]] = decision["selected_pose_id"]
    summary = read_json(output / "summary.json")
    summary.update(
        selected_pose_ids=selected,
        selected_case_count=len(selected),
        optimization_winner=selected or None,
        selection_implementation_sha256=sha256(Path(__file__)),
    )
    write_json(output / "summary.json", summary)
    return summary
