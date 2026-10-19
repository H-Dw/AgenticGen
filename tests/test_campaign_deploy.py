"""Verify source snapshots, configuration rendering and isolated deployment semantics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/deploy_ck2_campaign.py"
SPEC = importlib.util.spec_from_file_location("campaign_deploy", SCRIPT)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def local_snapshot(tmp_path, monkeypatch, content=b"print('snapshot')\n"):
    repo = tmp_path / "repo"
    module = repo / "src/agenticgen/campaign.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_bytes(content)
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    monkeypatch.setattr(deploy, "git_output", lambda *args: "a" * 40)
    archive = tmp_path / "source.tar.gz"
    return repo, archive, deploy.build_snapshot(repo, archive)


def install_payload(tmp_path, archive, manifest):
    root = tmp_path / "remote"
    (root / "environment").mkdir(parents=True)
    return {
        "root": str(root),
        "archive": str(archive),
        "archive_sha256": manifest["archive_sha256"],
        "snapshot_sha256": manifest["snapshot_sha256"],
        "external_paths": {},
        "configs": {"campaign.json": {"seed": 42}, "foundation.high.json": {"high": True}},
        "replace_code": False,
        "output": str(root / "results"),
    }


def remote_install(payload, prefix=""):
    return subprocess.run(
        [sys.executable, "-c", prefix + deploy.INSTALL_SOURCE],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def test_snapshot_preserves_bytes_excludes_outputs_and_credentials(tmp_path, monkeypatch):
    repo, archive, first = local_snapshot(tmp_path, monkeypatch)
    for name in ("scripts/.env", "docs/id_rsa", "examples/outputs/private.json", "src/x.pyc"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("never-copy-this")
    document = repo / "docs/protocol.md"
    document.write_bytes(b"# exact\r\n\xe4\xb8\xad\xe6\x96\x87\r\n")
    result = deploy.build_snapshot(repo, archive)
    assert result["snapshot_sha256"] != first["snapshot_sha256"]
    with tarfile.open(archive) as bundle:
        assert bundle.extractfile("docs/protocol.md").read() == document.read_bytes()
        assert not any(
            "never-copy-this" in bundle.extractfile(item).read().decode() for item in bundle
        )
    repeated = deploy.build_snapshot(repo, tmp_path / "again.tar.gz")
    assert repeated["snapshot_sha256"] == result["snapshot_sha256"]


def test_template_preserves_scientific_parameters_and_runtime_environment_reference():
    value = {
        "checkpoint": "@CHECKPOINT@",
        "seed": 20261020,
        "window": [0.1, 0.9],
        "nested": {"python": "@PYTHON@", "runtime": "${DEEPSEEK_API_KEY}"},
    }
    rendered = deploy.render_template(value, {"CHECKPOINT": "/fixed.ckpt", "PYTHON": "/env/python"})
    assert rendered == {
        "checkpoint": "/fixed.ckpt",
        "seed": 20261020,
        "window": [0.1, 0.9],
        "nested": {"python": "/env/python", "runtime": "${DEEPSEEK_API_KEY}"},
    }
    assert value["checkpoint"] == "@CHECKPOINT@"
    with pytest.raises(ValueError, match="Unknown deployment"):
        deploy.render_template("@UNKNOWN@", {})
    with pytest.raises(ValueError, match="Literal credential"):
        deploy.render_template({"llm": {"api_key": "a-secret-value"}}, {})


def test_install_verifies_hashes_and_leaves_existing_results_untouched(tmp_path, monkeypatch):
    _, archive, manifest = local_snapshot(tmp_path, monkeypatch)
    payload = install_payload(tmp_path, archive, manifest)
    results = Path(payload["output"])
    results.mkdir()
    output = results / "raw-particles.sdf"
    output.write_bytes(b"existing outcomes\n")
    result = remote_install(payload)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["formal_jobs_launched"] == 0
    assert receipt["snapshot_sha256"] == manifest["snapshot_sha256"]
    assert output.read_bytes() == b"existing outcomes\n"
    assert (
        Path(payload["root"]) / "code/src/agenticgen/campaign.py"
    ).read_bytes() == b"print('snapshot')\n"


def test_install_rejects_config_change_after_campaign_prepare(tmp_path, monkeypatch):
    _, archive, manifest = local_snapshot(tmp_path, monkeypatch)
    payload = install_payload(tmp_path, archive, manifest)
    assert remote_install(payload).returncode == 0
    results = Path(payload["output"])
    results.mkdir()
    (results / "manifest.json").write_text("{}")
    payload["configs"]["campaign.json"]["seed"] = 9
    result = remote_install(payload)
    assert result.returncode != 0 and "frozen configuration" in result.stderr
    assert json.loads((Path(payload["root"]) / "campaign.json").read_text())["seed"] == 42


def test_code_refresh_requires_explicit_flag_and_retains_previous_snapshot(tmp_path, monkeypatch):
    repo, archive, manifest = local_snapshot(tmp_path, monkeypatch)
    payload = install_payload(tmp_path, archive, manifest)
    assert remote_install(payload).returncode == 0
    (repo / "src/agenticgen/campaign.py").write_bytes(b"print('updated source')\n")
    updated = deploy.build_snapshot(repo, archive)
    payload.update(
        archive_sha256=updated["archive_sha256"], snapshot_sha256=updated["snapshot_sha256"]
    )
    rejected = remote_install(payload)
    assert rejected.returncode != 0 and "--replace-code" in rejected.stderr
    assert not list(Path(payload["root"]).glob("code.staging.*"))
    payload["replace_code"] = True
    result = remote_install(payload)
    assert result.returncode == 0, result.stderr
    previous = Path(json.loads(result.stdout)["previous_code"])
    assert (previous / "src/agenticgen/campaign.py").read_bytes() == b"print('snapshot')\n"
    assert (
        Path(payload["root"]) / "code/src/agenticgen/campaign.py"
    ).read_bytes() == b"print('updated source')\n"


def test_install_refuses_code_replacement_while_controller_is_active(tmp_path, monkeypatch):
    _, archive, manifest = local_snapshot(tmp_path, monkeypatch)
    payload = install_payload(tmp_path, archive, manifest)
    jobs = Path(payload["root"]) / "environment/jobs"
    jobs.mkdir()
    (jobs / "test.launch.json").write_text(json.dumps({"pid": 12345}))
    # Model Linux kill(pid, 0) success without sending process signals on Windows.
    result = remote_install(payload, "import os\nos.kill = lambda pid, signal: None\n")
    assert result.returncode != 0 and "controller is active" in result.stderr
    assert not (Path(payload["root"]) / "code").exists()


def test_sibling_release_does_not_touch_active_code_and_promotes_identical_bytes(
    tmp_path, monkeypatch
):
    _, archive, manifest = local_snapshot(tmp_path, monkeypatch)
    payload = install_payload(tmp_path, archive, manifest)
    assert remote_install(payload).returncode == 0
    root = Path(payload["root"])
    active_code = root / "code"
    before = {
        p.relative_to(active_code).as_posix(): p.read_bytes()
        for p in active_code.rglob("*")
        if p.is_file()
    }
    jobs = root / "environment/jobs"
    jobs.mkdir()
    (jobs / "design.launch.json").write_text(json.dumps({"pid": 12345}))
    process = subprocess.run(
        [sys.executable, "-c", deploy.RELEASE_SOURCE],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    receipt = json.loads(process.stdout)
    assert receipt["active_code_unchanged"] is True
    assert {
        p.relative_to(active_code).as_posix(): p.read_bytes()
        for p in active_code.rglob("*")
        if p.is_file()
    } == before
    release_code = Path(receipt["code"])
    released = {
        p.relative_to(release_code).as_posix(): p.read_bytes()
        for p in release_code.rglob("*")
        if p.is_file()
    }
    (jobs / "design.status.json").write_text(json.dumps({"status": "completed"}))

    class LocalRemote:
        @staticmethod
        def python(source, value):
            completed = subprocess.run(
                [sys.executable, "-c", source],
                input=json.dumps(value),
                text=True,
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            return json.loads(completed.stdout)

    args = deploy.parser().parse_args(
        [
            "promote",
            "--remote-root",
            str(root).replace("\\", "/"),
            "--release",
            manifest["snapshot_sha256"],
        ]
    )
    promoted = deploy.promote(
        LocalRemote(), args, {"RUN_ROOT": str(root), "OUTPUT_ROOT": str(root / "results")}
    )
    assert promoted["snapshot_sha256"] == manifest["snapshot_sha256"]
    assert {
        p.relative_to(active_code).as_posix(): p.read_bytes()
        for p in active_code.rglob("*")
        if p.is_file()
    } == released


def test_launch_is_explicit_uses_interactive_key_loading_and_sequential_phases():
    args = deploy.parser().parse_args(["launch", "--phase", "baseline-design"])
    paths = deploy.remote_paths(args)
    payload = deploy.launch_payload(args, paths)
    assert [command["phase"] for command in payload["commands"]] == ["baseline", "design"]
    assert "conda activate molsteer-flowr-dtk" in payload["shell"]
    assert "http://127.0.0.1:17897" in payload["shell"]
    assert "DEEPSEEK_API_KEY" not in payload["shell"]
    assert all(command["argv"][0] == deploy.DEFAULT_PYTHON for command in payload["commands"])
    assert '"bash","-ic"' in deploy.LAUNCH_SOURCE
    with pytest.raises(ValueError, match="explicit --phase"):
        deploy.main(["launch"])


def test_existing_upstream_is_not_a_valid_experiment_root():
    args = deploy.parser().parse_args(["deploy", "--remote-root", "/workspace/AgenticGen"])
    with pytest.raises(ValueError, match="distinct absolute"):
        deploy.remote_paths(args)
