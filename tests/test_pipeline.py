import json
import shutil
from dataclasses import replace

import pytest

from agenticgen.config import load_configuration
from agenticgen.dataset import load_dataset
from agenticgen.errors import ArtifactError
from agenticgen.factory import build_components
from agenticgen.io import read_json
from agenticgen.runner import run


def test_ck2_identity_and_original_pose_integrity(repo, tmp_path):
    folder = repo / "examples/ck2_top3"
    dataset = load_dataset(folder / "cases.json")
    assert [c.case_id for c in dataset.cases] == [
        "single_b004_final_027",
        "single_b017_final_021",
        "single_b007_final_026",
    ]
    assert all(c.source_metrics["quality_tail"] for c in dataset.cases)
    assert [c.source_metrics["pic50_on_rescore"] for c in dataset.cases] == sorted(
        [c.source_metrics["pic50_on_rescore"] for c in dataset.cases], reverse=True
    )
    shutil.copy(folder / "cases.json", tmp_path)
    shutil.copy(folder / "candidates.sdf", tmp_path)
    with (tmp_path / "candidates.sdf").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ArtifactError, match="checksum"):
        load_dataset(tmp_path / "cases.json")


@pytest.mark.integration
def test_real_syncraft_replay_pipeline_has_no_fabricated_route_or_geometry(
    repo, tmp_path, syncraft_root
):
    config = load_configuration(repo / "examples/ck2_top3/config.json", "replay")
    config = replace(
        config,
        value=config.value.model_copy(
            update={
                "syncraft": config.value.syncraft.model_copy(update={"root": str(syncraft_root)})
            }
        ),
    )
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_components(config, dataset)
    summary = run(config, dataset, components, tmp_path / "run")
    assert summary["accepted_edit_count"] == 3
    assert summary["primary_route_evaluated_count"] == 0
    assert summary["binding_and_geometry_evaluated"] is False
    assert summary["llm_backend"] == "replay"
    for case in dataset.cases:
        output = tmp_path / "run/cases" / case.case_id
        result = read_json(output / "result.json")
        assert result["candidates"][0]["retrosynthesis"]["status"] == "not_run"
        assert result["candidates"][0]["constrained_route_success"] is None
        assert "retrosynthesis" not in read_json(output / "decision.json")["proposals"][0]
        observation = read_json(output / "observation.json")
        assert "quality_tail" not in observation["terminal_metrics"]
        assert "selection_rank" not in observation
    with pytest.raises(FileExistsError):
        run(config, dataset, components, tmp_path / "run")


@pytest.mark.integration
def test_failed_agent_is_partial_and_has_no_substituted_candidate(repo, tmp_path, syncraft_root):
    config = load_configuration(repo / "examples/ck2_top3/config.json", "replay")
    config = replace(
        config,
        value=config.value.model_copy(
            update={
                "syncraft": config.value.syncraft.model_copy(update={"root": str(syncraft_root)})
            }
        ),
    )
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_components(config, dataset)

    class BrokenAgent:
        def propose(self, *args):
            raise RuntimeError("Intentional interface failure")

    summary = run(config, dataset, replace(components, agent=BrokenAgent()), tmp_path / "failed")
    assert summary["interface_error_count"] == 3
    assert summary["accepted_edit_count"] == 0
    assert read_json(tmp_path / "failed/manifest.json")["status"] == "partial"
    assert not (tmp_path / "failed/products.smi").read_text()


def test_t_only_config_rejects_unimplemented_experimental_arms(repo, tmp_path):
    from pydantic import ValidationError

    data = read_json(repo / "examples/ck2_top3/config.json")
    data["arm"] = "TH"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_configuration(path)
