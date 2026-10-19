from dataclasses import replace
from pathlib import Path

from test_foundation import configuration

from agenticgen.base_factory import build_foundation
from agenticgen.contracts import GraphEditResult
from agenticgen.dataset import load_dataset
from agenticgen.design_runner import design_run
from agenticgen.inpainting_contracts import InpaintingConfig
from agenticgen.inpainting_runner import inpainting_run
from agenticgen.io import read_json, write_json


class ForbiddenEvaluator:
    def evaluate(self, *args):
        raise AssertionError("Product property/route evaluator ran before inpainting")


def test_design_stage_never_scores_edited_products(repo, tmp_path, syncraft_root):
    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_foundation(config, dataset)
    components = replace(components, properties=ForbiddenEvaluator(), retro=ForbiddenEvaluator())
    folder = tmp_path / "design"
    summary = design_run(config, dataset, components, folder)
    assert summary["accepted_graph_count"] == 5
    assert summary["product_evaluations"] == 0
    assert summary["optimization_winner"] is None
    result = read_json(folder / "cases" / dataset.cases[0].case_id / "result.json")
    assert all("properties" not in c and "binding" not in c for c in result["candidates"])


def test_invalid_graph_skips_before_any_pose_or_flowr_preparation(repo, tmp_path):
    source = tmp_path / "design"
    folder = source / "cases/test"
    write_json(source / "manifest.json", {"schema_version": "agenticgen.design-run.v1"})
    # Use the actual versioned contract without providing any pose/model/runtime files.
    from agenticgen.contracts import EditConstraints, Observation

    observation = Observation(
        case_id="test",
        target="CK2",
        source_smiles="CC",
        mapped_smiles="[CH3:1][CH3:2]",
        graph_sha256="fixture",
        terminal_metrics={},
        constraints=EditConstraints(),
    )
    write_json(folder / "observation.json", observation.model_dump(mode="json"))
    graph = GraphEditResult(status="rejected", source_smiles="CC", reason="Invalid valence")
    write_json(
        folder / "result.json",
        {
            "case_id": "test",
            "source_binding": {
                "status": "not_run",
                "protocol_id": "disabled",
                "graph_smiles": "CC",
            },
            "candidates": [
                {"proposal_index": 0, "status": "rejected", "graph": graph.model_dump(mode="json")}
            ],
        },
    )
    config = InpaintingConfig(root="missing", python="missing", checkpoint="missing")
    output = tmp_path / "inpaint"
    result = inpainting_run(source, config, Path("absent-pocket.pdb"), output, execute=True)
    assert result["skipped_proposal_count"] == 1
    assert result["failed_proposal_count"] == result["measured_pose_count"] == 0
    saved = read_json(output / "cases/test/proposal_00/inpainting/result.json")
    assert all(mode["status"] == "skipped" for mode in saved["mode_results"])
    assert not (output / "cases/test/proposal_00/initialized_product.sdf").exists()
