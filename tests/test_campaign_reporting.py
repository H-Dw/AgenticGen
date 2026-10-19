"""Coverage and protocol reports must not turn missing/obsolete data into results."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agenticgen.campaign_contracts import CampaignConfig
from agenticgen.campaign_ledger import JobLedger
from agenticgen.campaign_preflight import _check_trace, run_preflight
from agenticgen.campaign_report import report_campaign
from agenticgen.io import read_json, sha256, write_json


def config():
    return CampaignConfig(
        foundation_config="foundation.json",
        flowr_root="flowr",
        flowr_python="python",
        checkpoint="model.ckpt",
        pocket_pdb="pocket.pdb",
    )


def commit(campaign, identity, stage, result, payload=None, case_id="case_a", *, skipped=False):
    folder = campaign.root / "fixtures" / identity.replace(":", "_")
    campaign.ledger.add(identity, stage, case_id, payload or {}, folder)
    campaign.ledger.claim(identity)
    path = folder / "result.json"
    write_json(path, result)
    campaign.ledger.finish(identity, path, skipped=skipped)
    return path


def setup_report(tmp_path, *, evaluate=True, recovery=False):
    cases = [
        SimpleNamespace(
            case_id=f"case_{name}",
            selection_rank=index + 1,
            smiles="CCO",
            source_metrics={"pic50_on_rescore": 8.9 - index / 10},
        )
        for index, name in enumerate(("a", "b"))
    ]
    campaign = SimpleNamespace(
        root=tmp_path,
        config=config(),
        dataset=SimpleNamespace(cases=cases),
        cases={case.case_id: case for case in cases},
        ledger=JobLedger(tmp_path / "ledger.sqlite"),
    )
    commit(
        campaign,
        "source_geometry:case_a",
        "source_geometry",
        {
            "binding": {"score_kcal_mol": -8.0, "interactions": [{"kind": "HydrogenBond"}]},
            "properties": {"value": 3.5},
        },
    )
    commit(
        campaign,
        "source_affinity:all",
        "source_affinity",
        {"poses": [{"pose_id": "case_a", "status": "ok", "pic50": 7.0}]},
        case_id="all",
    )
    commit(
        campaign,
        "design:case_a",
        "design",
        {
            "status": "accepted",
            "chemical_proposal_count": 2,
            "retry_count": 1,
            "graph": {"product_smiles": "CO"},
            "properties": {"value": 2.5},
            "attempts": [
                {
                    "attempt_index": 0,
                    "status": "rejected",
                    "graph": {"reason": "Non-ring aromatic atom", "error_stage": "product_loading"},
                },
                {
                    "attempt_index": 1,
                    "status": "accepted",
                    "proposal": {"rationale": "A test edit; no real API called"},
                    "graph": {"product_smiles": "CO"},
                },
            ],
        },
    )
    commit(
        campaign,
        "design:case_b",
        "design",
        {"status": "abstained", "chemical_proposal_count": 1, "retry_count": 0},
        case_id="case_b",
    )
    particles = [
        {
            "pose_id": f"p{index}",
            "particle_index": index,
            "status": "decode_failed" if index == 2 else "decoded",
            "smiles": None if index == 2 else "CO" if index == 3 else "CCO",
        }
        for index in range(4)
    ]
    folder = tmp_path / "current_generation"
    write_json(folder / "poses.json", {"sample_count": 4, "particles": particles})
    write_json(
        folder / "affinity.json",
        {
            "status": "failed" if recovery else "completed",
            "poses": [
                {
                    "pose_id": f"p{i}",
                    "status": "unavailable" if i == 2 else "error" if recovery else "ok",
                    "pic50": None if i == 2 else 666.0 if recovery else 7.4,
                }
                for i in range(4)
            ],
        },
    )
    generation_id = "generation:case_a:graph:batch_000"
    result = commit(
        campaign,
        generation_id,
        "generation",
        {"output_dir": str(folder), "attempted": 4},
        {"arm_id": "graph", "batch_index": 0, "seed": 77},
    )
    parent = {"generation_task_id": generation_id, "generation_result_sha256": sha256(result)}
    if evaluate:
        rows = [
            {
                "pose_id": f"p{index}",
                "status": "unavailable" if index == 2 else "measured",
                "affinity": {
                    "status": "unavailable" if index == 2 else "ok",
                    "pic50": None if index == 2 else 7.4,
                },
                "binding": {"score_kcal_mol": None if index == 2 else -7.8},
                "delta_vina_score_only": None if index == 2 else 0.2,
            }
            for index in range(4)
        ]
        commit(campaign, "evaluation:current", "evaluation", {"particles": rows}, parent)
    if recovery:
        scores = [
            {
                "pose_id": f"p{i}",
                "status": "unavailable" if i == 2 else "ok",
                "pic50": None if i == 2 else 7.8,
            }
            for i in range(4)
        ]
        commit(
            campaign,
            "affinity:recovered",
            "affinity",
            {"status": "completed", "poses": scores},
            parent,
        )
    return campaign, parent


def table(campaign, name):
    with (campaign.root / "report" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report_retains_invalid_and_duplicate_attempts_and_separates_baseline_protocol(tmp_path):
    campaign, _ = setup_report(tmp_path)
    result = report_campaign(campaign)
    assert result["observed_generation_attempts"] == 4
    assert result["decoded_particles"] == 3
    sources = {row["case_id"]: row for row in table(campaign, "sources.csv")}
    assert float(sources["case_a"]["historical_selection_pic50"]) == 8.9
    assert float(sources["case_a"]["common_protocol_pic50"]) == 7.0
    assert sources["case_b"]["common_protocol_pic50"] == ""
    assert sources["case_b"]["original_contact_count"] == ""
    assert sources["case_b"]["baseline_status"] == "pending"
    assert sources["case_a"]["chemical_proposal_count"] == "2"
    assert sources["case_a"]["retries"] == "1"
    proposals = table(campaign, "proposals.csv")
    assert proposals[0]["reason"] == "Non-ring aromatic atom"
    assert proposals[0]["error_stage"] == "product_loading"
    particles = table(campaign, "particles.csv")
    assert len(particles) == 4
    assert particles[2]["decode_status"] == "decode_failed"
    assert particles[2]["pic50"] == ""
    assert all(float(row["delta_pic50"]) == pytest.approx(0.4) for row in particles if row["pic50"])
    arms = {row["arm_id"]: row for row in table(campaign, "arms.csv")}
    assert arms["graph"]["planned_attempts"] == "100"
    assert arms["graph"]["observed_attempts"] == "4"
    assert float(arms["graph"]["decode_fraction_of_observed_attempts"]) == 0.75
    assert arms["graph"]["unique_graphs"] == "2"
    assert arms["graph"]["duplicates"] == "1"
    assert arms["graph"]["affinity_scored"] == "3"


def test_report_excludes_obsolete_parent_and_superseded_measurements(tmp_path):
    campaign, parent = setup_report(tmp_path)
    stale = {
        "particles": [
            {"pose_id": f"p{i}", "affinity": {"status": "ok", "pic50": 999.0}} for i in range(4)
        ]
    }
    # Even if an old controller did not mark this child superseded, its parent
    # digest is obsolete and must not contaminate the current generation.
    commit(
        campaign,
        "evaluation:obsolete_parent",
        "evaluation",
        stale,
        {**parent, "generation_result_sha256": "old-execution"},
    )
    commit(campaign, "evaluation:superseded", "evaluation", stale, parent)
    campaign.ledger.supersede("evaluation:superseded", "New head measurement")
    report_campaign(campaign)
    values = [float(row["pic50"]) for row in table(campaign, "particles.csv") if row["pic50"]]
    assert values == [7.4, 7.4, 7.4]


@pytest.mark.parametrize("old_evaluation_present", [False, True])
def test_report_uses_recovered_current_head_before_updated_geometry_evaluation(
    tmp_path, old_evaluation_present
):
    campaign, _ = setup_report(tmp_path, evaluate=old_evaluation_present, recovery=True)
    report_campaign(campaign)
    rows = table(campaign, "particles.csv")
    assert [float(row["pic50"]) for row in rows if row["pic50"]] == [7.8, 7.8, 7.8]
    if not old_evaluation_present:
        assert all(row["evaluation_status"] == "pending" for row in rows)
    assert rows[2]["pic50"] == ""


def test_report_distinguishes_not_applicable_arm_from_unstarted_work(tmp_path):
    campaign, _ = setup_report(tmp_path)
    for index in range(25):
        commit(
            campaign,
            f"generation:case_a:fragment_graph_pose:batch_{index:03d}",
            "generation",
            {
                "status": "not_applicable",
                "reason": "No original interaction anchors",
                "attempted": 0,
            },
            {"arm_id": "fragment_graph_pose", "batch_index": index, "not_applicable": True},
            skipped=True,
        )
    report_campaign(campaign)
    arms = {row["arm_id"]: row for row in table(campaign, "arms.csv")}
    assert arms["fragment_graph_pose"]["planned_attempts"] == "0"
    assert arms["fragment_graph_pose"]["applicability"] == "not_applicable"
    assert arms["fragment_graph_pose"]["observed_attempts"] == "0"
    assert arms["fragment_graph"]["planned_attempts"] == "100"
    assert arms["fragment_graph"]["observed_attempts"] == "0"


def trace_rows():
    return [
        {
            "time": round(index / 100, 2),
            "logit_gradient_rms": [0.1],
            "coordinate_gradient_rms": [0.2],
            "injected_rms_angstrom": [0.01],
            "injected_max_atom_angstrom": [0.025],
        }
        for index in range(10, 90)
    ]


def test_reporting_module_import_does_not_require_optional_torch():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Reporting must not import optional Torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import agenticgen.campaign_report
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_preflight_trace_rejects_nonfinite_producer_output(tmp_path):
    rows = trace_rows()
    rows[5]["logit_gradient_rms"] = [float("nan")]
    # A corrupt external producer can bypass AgenticGen's normal finite-JSON writer.
    (tmp_path / "guidance_trace.json").write_text(json.dumps(rows), encoding="utf-8")
    write_json(
        tmp_path / "guidance_probe.json", {"status": "completed", "report": {"passed": True}}
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        _check_trace(tmp_path)


@pytest.mark.parametrize("failure", ["clock", "inactive", "cap", "derivative"])
def test_preflight_trace_rejects_wrong_clock_inactive_gradient_caps_and_failed_derivative(
    tmp_path, failure
):
    rows = trace_rows()
    probe = {"status": "completed", "report": {"passed": True}}
    if failure == "clock":
        rows[-1]["time"] = 0.9
    elif failure == "inactive":
        for row in rows:
            row["coordinate_gradient_rms"] = [0.0]
    elif failure == "cap":
        rows[0]["injected_max_atom_angstrom"] = [0.026]
    else:
        probe["report"]["passed"] = False
    write_json(tmp_path / "guidance_trace.json", rows)
    write_json(tmp_path / "guidance_probe.json", probe)
    with pytest.raises(RuntimeError):
        _check_trace(tmp_path)


@pytest.mark.parametrize("mismatch", [None, "prior", "rng", "zero", "world"])
def test_real_preflight_contract_world_targets_and_matched_native_controls(tmp_path, mismatch):
    """A CPU tensor worker double supplies independently framed reference/output tensors."""
    torch = pytest.importorskip("torch")
    conditions_path = tmp_path / "conditions.json"
    conditions = {
        "status": "ready",
        "atom_maps": [1, 2, 3],
        "fixed_atom_input_slots": [0, 1],
        "fixed_bond_pairs": [[0, 0], [0, 1], [1, 1]],
        "interaction_input_slots": [0],
        "interaction_coordinates_world": [[12.0, -3.0, 4.0]],
    }
    write_json(conditions_path, conditions)
    design_path = tmp_path / "design.json"
    write_json(
        design_path, {"conditions_file": str(conditions_path), "product_pose_sdf": "product.sdf"}
    )

    def worker(module, task, folder, timeout):
        if module == "agenticgen.campaign_affinity":
            write_json(
                folder / "affinity.json",
                {
                    "status": "completed",
                    "poses": [{"status": "ok", "pic50": 7.0} for _ in task["poses"]],
                },
            )
            return
        label = folder.name
        reference = {
            "atomics": torch.nn.functional.one_hot(torch.tensor([[0, 1, 0]] * 4), 2).float(),
            "charges": torch.ones((4, 3, 1)),
            "bonds": torch.nn.functional.one_hot(
                torch.zeros((4, 3, 3), dtype=torch.long), 2
            ).float(),
            "coords": torch.full((4, 3, 3), -999.0),  # normalized; deliberately not world anchors
            "mask": torch.ones((4, 3), dtype=torch.bool),
        }
        raw = {key: value.clone() for key, value in reference.items()}
        raw["coords"].fill_(0.5)
        if label in {"ce_graph", "brier"}:
            raw["coords"] += 0.01
        if label == "fragment_graph_pose":
            raw["coords"][:, 0] = torch.tensor(conditions["interaction_coordinates_world"][0])
            if mismatch == "world":
                raw["coords"][:, 0] = reference["coords"][:, 0]
        if mismatch == "zero" and label == "zero_guidance":
            raw["coords"] += 0.001
        torch.save(raw, folder / "raw_final.pt")
        torch.save(reference, folder / "raw_reference.pt")
        write_json(
            folder / "poses.json",
            {
                "schema_version": "agenticgen.flowr-particles.v1",
                "sample_count": 4,
                "particles": [{"status": "decode_failed"}] * 4,
            },
        )
        runtime = {
            "status": "completed",
            "generation_completed": True,
            "decoded_count": 0,
            "base_prior_sha256": "common-prior",
            "rng_before_generation_sha256": "common-rng",
            "final_tensor_sha256": "guided" if label in {"ce_graph", "brier"} else "native",
        }
        if mismatch == "prior" and label == "brier":
            runtime["base_prior_sha256"] = "different-prior"
        if mismatch == "rng" and label == "brier":
            runtime["rng_before_generation_sha256"] = "different-rng"
        write_json(folder / "runtime.json", runtime)
        if task["guidance_probe"]:
            write_json(folder / "guidance_trace.json", trace_rows())
            write_json(
                folder / "guidance_probe.json", {"status": "completed", "report": {"passed": True}}
            )

    campaign = SimpleNamespace(
        config=config(),
        source_path=lambda case_id: tmp_path / "source.sdf",
        _flowr_task=lambda folder, seed: {"output_dir": str(folder), "seed": seed},
        _worker=worker,
    )
    output = tmp_path / "preflight"
    if mismatch is None:
        report = run_preflight(
            campaign, {"result_path": str(design_path), "case_id": "case_a"}, output
        )
        assert report["status"] == "passed"
        assert report["production_particles"] == 0
        assert report["checks"]["fragment_graph_pose"]["terminal_conditions_checked"]
    else:
        with pytest.raises(RuntimeError):
            run_preflight(campaign, {"result_path": str(design_path), "case_id": "case_a"}, output)
        assert read_json(output / "result.json")["status"] == "failed"
