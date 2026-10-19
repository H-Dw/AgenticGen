import json
import subprocess
from types import SimpleNamespace

import pytest

from agenticgen.adapters.retro import SimpRetroCLI


@pytest.fixture
def retro(tmp_path):
    root = tmp_path / "external"
    (root / "syntheseus/cli").mkdir(parents=True)
    (root / "syntheseus/cli/search.py").write_text("# Test-only CLI placeholder")
    for name in ("python", "inventory", "templates"):
        (root / name).write_text("test-only")
    return SimpRetroCLI(
        root=root,
        python=root / "python",
        inventory=root / "inventory",
        templates=root / "templates",
        time_limit_seconds=10,
    )


@pytest.mark.parametrize(
    "solution, expected", [(2.5, "route_found"), (float("inf"), "no_route_within_budget")]
)
def test_search_statistics_are_bound_to_input_and_never_route_certificates(
    retro, tmp_path, monkeypatch, solution, expected
):
    def fake_search(command, **kwargs):
        output = tmp_path / "job/search/test"
        output.mkdir(parents=True)
        (output / "stats.json").write_text(
            json.dumps({"smiles": "CCO", "soln_time_wallclock": solution})
        )
        assert "seed=0" not in command  # upstream fixes the seed internally; no invented flag
        assert "use_gpu=false" in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_search)
    result = retro.evaluate("CCO", tmp_path / "job")
    assert result.status == expected
    assert result.route_verified is False


def test_timeout_is_infrastructure_error_not_unreachable(retro, tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("unit-test", 10)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert retro.evaluate("CCO", tmp_path / "timeout").status == "infrastructure_error"


def test_docker_search_and_validation_translate_paths_and_fix_stock(retro, tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="sha256:test-image\n"),
    )
    worker = SimpRetroCLI(
        root=retro.root,
        python=None,
        inventory=retro.inventory,
        templates=retro.templates,
        execution="docker",
        image="test-image",
    )
    output = tmp_path / "job"
    command = worker.command(output / "targets.smi", output / "search")
    assert command[:3] == ["docker", "run", "--rm"]
    assert "inventory_smiles_file=/job/runtime/emolecules.txt" in command
    assert "search_targets_file=/job/targets.smi" in command
    assert "results_dir=/job/search" in command
    checked = worker.validation_command(output / "search/target/graph.pkl", output, "CCO")
    assert checked[-5:] == [
        "/job/search/target/graph.pkl",
        "/job/runtime/emolecules.txt",
        "/templates/templates.json",
        "CCO",
        "/job/route.json",
    ]
