import pytest

from agenticgen.cli import main
from agenticgen.contracts import RetroResult
from agenticgen.io import read_json
from agenticgen.retro_screen import screen_targets


def test_retro_screen_preserves_target_order_and_does_not_conflate_infrastructure(repo, tmp_path):
    targets = tmp_path / "targets.smi"
    targets.write_text("CCO\nCCN\nCCC\n", encoding="utf-8")

    class Evaluator:
        def evaluate(self, smiles, output):
            status = {
                "CCO": "route_found",
                "CCN": "no_route_within_budget",
                "CCC": "infrastructure_error",
            }[smiles]
            return RetroResult(status=status, protocol_id="test", route_verified=False)

    output = tmp_path / "screen"
    summary = screen_targets(targets, output, Evaluator(), {"fixture": True})
    assert summary["verified_route_count"] == 0
    assert summary["status_counts"] == {
        "route_found": 1,
        "no_route_within_budget": 1,
        "infrastructure_error": 1,
    }
    assert read_json(output / "manifest.json")["status"] == "partial"
    assert [r["smiles"] for r in read_json(output / "results.json")] == ["CCO", "CCN", "CCC"]
    targets.write_text("CCO\nOCC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        screen_targets(targets, tmp_path / "duplicate", Evaluator(), {})
    assert (
        main(
            [
                "screen-retro",
                "--config",
                str(repo / "examples/ck2_top3/config.json"),
                "--targets",
                str(targets),
                "--output",
                str(tmp_path / "disabled"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "disabled").exists()
