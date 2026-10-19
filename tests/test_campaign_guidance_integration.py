"""Protocol-level integration tests without a network, docking binary, or GPU."""

from __future__ import annotations

import random
import sys
from collections import Counter
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from agenticgen.campaign import Campaign, particle_rows
from agenticgen.campaign_contracts import CampaignConfig, campaign_arms
from agenticgen.campaign_ledger import JobLedger
from agenticgen.contracts import (
    BindingResult,
    EditConstraints,
    GraphEditResult,
    Interaction,
    Proposal,
)
from agenticgen.io import object_hash, read_json, write_json


def campaign_fixture(tmp_path):
    campaign = Campaign.__new__(Campaign)
    campaign.root = tmp_path / "run"
    campaign.config_path = tmp_path / "config.json"
    campaign.config = CampaignConfig(
        foundation_config="foundation.json",
        flowr_root="flowr",
        flowr_python="python",
        checkpoint="model.ckpt",
        pocket_pdb="pocket.pdb",
    )
    campaign.ledger = JobLedger(campaign.root / "ledger.sqlite")
    campaign.cases = {
        "case_a": SimpleNamespace(case_id="case_a", target="CK2", smiles="CCO", source_metrics={})
    }
    campaign.foundation = SimpleNamespace(value=SimpleNamespace(constraints=EditConstraints()))
    campaign._components = None
    campaign._gpu_session = None
    return campaign


def commit_job(campaign, task_id, stage, payload, result, folder):
    campaign.ledger.add(task_id, stage, "case_a", payload, folder)
    campaign.ledger.claim(task_id)
    path = folder / "result.json"
    write_json(path, result)
    campaign.ledger.finish(task_id, path)


def test_accepted_graph_arms_have_exactly_100_attempts_and_paired_seeds(tmp_path):
    campaign = campaign_fixture(tmp_path)
    conditions_path = tmp_path / "conditions.json"
    write_json(
        conditions_path,
        {
            "status": "ready",
            "atom_maps": [1, 2, 3],
            "fixed_atom_input_slots": [0, 1],
            "fixed_bond_pairs": [[0, 0], [0, 1], [1, 1], [1, 2]],
            "interaction_input_slots": [0],
            "interaction_coordinates_world": [[1.0, 2.0, 3.0]],
        },
    )
    commit_job(
        campaign,
        "design:case_a",
        "design",
        {},
        {
            "status": "accepted",
            "product_pose_sdf": str(tmp_path / "product.sdf"),
            "conditions_file": str(conditions_path),
        },
        tmp_path / "design",
    )
    campaign.enqueue_generation()
    jobs = campaign.ledger.list(stage="generation")
    assert len(campaign_arms(campaign.config)) == 16
    assert len(jobs) == 16 * 25
    counts = Counter()
    seeds_by_batch = {}
    for job in jobs:
        payload = job["payload"]
        counts[payload["arm_id"]] += payload["sample_count"]
        seeds_by_batch.setdefault(payload["batch_index"], set()).add(payload["seed"])
        assert payload["integration_steps"] == 100
        assert payload["guidance"]["window"] == [0.1, 0.9]
        assert payload["source_pose_sdf"] == str(campaign.source_path("case_a"))
    assert set(counts.values()) == {100}
    assert sum(counts.values()) == 1600
    assert all(len(seeds) == 1 for seeds in seeds_by_batch.values())
    assert len({next(iter(seeds)) for seeds in seeds_by_batch.values()}) == 25
    campaign.enqueue_generation()
    assert len(campaign.ledger.list(stage="generation")) == 400
    assert all(job["attempts"] == 0 for job in campaign.ledger.list(stage="generation"))


@pytest.mark.parametrize("status", ["abstained", "unchanged", "chemical_retries_exhausted"])
def test_nonusable_design_outcome_never_enqueues_generator(tmp_path, status):
    campaign = campaign_fixture(tmp_path)
    commit_job(
        campaign,
        "design:case_a",
        "design",
        {},
        {"status": status},
        tmp_path / "design",
    )
    assert campaign.enqueue_generation() == 0
    assert campaign.ledger.list(stage="generation") == []


def original_binding():
    return BindingResult(
        status="ok",
        protocol_id="test-original-pose",
        graph_smiles="CCO",
        prompt="Measured original Steer contacts",
        evaluation_mode="score_only",
        score_kcal_mol=-8.0,
    )


def test_design_uses_durable_api_response_without_duplicate_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "engineering-test-fixture-no-network")
    campaign = campaign_fixture(tmp_path)
    campaign._source_binding = lambda case_id: original_binding()
    parse_calls = []

    def parse(text):
        parse_calls.append(text)
        return {"reasoning": "No graph edit proposed", "edits": [["STOP"]]}

    def unexpected_call(*args, **kwargs):
        raise AssertionError("A durable API response must not be requested again")

    campaign._components = SimpleNamespace(
        agent=SimpleNamespace(
            backend=SimpleNamespace(inference=SimpleNamespace(parse_llm_output=parse)),
            propose=unexpected_call,
        )
    )
    attempt = campaign.source_path("case_a").parent / "proposals/attempt_00/transport_000"
    write_json(attempt / "response.json", {"native_output": "durable completed response"})
    first = campaign._design({"case_id": "case_a"}, tmp_path / "execution_a")
    second = campaign._design({"case_id": "case_a"}, tmp_path / "execution_b")
    assert first["status"] == second["status"] == "abstained"
    assert first["chemical_proposal_count"] == 1
    assert parse_calls == ["durable completed response"]


def test_unparsable_graph_has_five_extra_retries_and_resume_reuses_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "engineering-test-fixture-no-network")
    campaign = campaign_fixture(tmp_path)
    campaign._source_binding = lambda case_id: original_binding()
    calls = []

    def propose(observation, binding, index, output):
        calls.append(index)
        assert binding.prompt == "Measured original Steer contacts"
        write_json(output, {"native_output": f"fake graph proposal {index}"})
        return Proposal(
            case_id=observation.case_id,
            action="edit",
            edits=[["REMOVE_BOND", 1, 2]],
            rationale="Test malformed graph without a real API",
            expected_effect="Unknown until evaluated",
        )

    def reject(*args):
        return GraphEditResult(
            status="rejected",
            source_smiles="CCO",
            error_stage="product_loading",
            reason="Native product cannot be parsed",
        )

    campaign._components = SimpleNamespace(
        agent=SimpleNamespace(propose=propose), editor=SimpleNamespace(apply=reject)
    )
    result = campaign._design({"case_id": "case_a"}, tmp_path / "execution_a")
    assert calls == list(range(6))
    assert result["chemical_proposal_count"] == 6
    assert result["retry_count"] == 5
    assert result["status"] == "chemical_retries_exhausted"
    assert all(not attempt["prompt_changed_for_retry"] for attempt in result["attempts"])
    repeated = campaign._design({"case_id": "case_a"}, tmp_path / "execution_b")
    assert calls == list(range(6))
    assert repeated["chemical_proposal_count"] == 6


def test_source_geometry_uses_actual_pose_and_score_only(tmp_path, monkeypatch):
    campaign = campaign_fixture(tmp_path)
    path = campaign.source_path("case_a")
    path.parent.mkdir(parents=True)
    path.write_text("Original pose fixture", encoding="utf-8")
    calls = []

    def evaluate_pose(smiles, pose, output, *, mode):
        calls.append((pose, mode))
        return original_binding()

    campaign._components = SimpleNamespace(
        geometry=SimpleNamespace(
            evaluate_pose=evaluate_pose, files={"receptor_pdb": tmp_path / "receptor.pdb"}
        ),
        properties=SimpleNamespace(
            evaluate=lambda smiles: SimpleNamespace(model_dump=lambda **kwargs: {"status": "ok"})
        ),
    )
    monkeypatch.setattr("agenticgen.campaign.measure_pose", lambda path: {"status": "fake"})
    monkeypatch.setattr("agenticgen.campaign.posebusters", lambda *args: {"status": "fake"})
    result = campaign._source_geometry({"case_id": "case_a"}, tmp_path / "eval")
    assert calls == [(path, "score_only")]
    assert result["binding"]["evaluation_mode"] == "score_only"
    assert path.read_text(encoding="utf-8") == "Original pose fixture"


def test_particle_receipt_keeps_failed_attempts_and_rejects_partial_denominator(tmp_path):
    path = tmp_path / "poses.json"
    rows = [{"pose_id": f"p{i}", "status": "decode_failed"} for i in range(4)]
    write_json(
        path,
        {"schema_version": "agenticgen.flowr-particles.v1", "sample_count": 4, "particles": rows},
    )
    assert particle_rows(path) == rows
    value = read_json(path)
    value["particles"].pop()
    write_json(path, value)
    with pytest.raises(ValueError, match="accounting is incomplete"):
        particle_rows(path)


def test_ledger_atomic_claim_and_immutable_inputs_prevent_duplicate_execution(tmp_path):
    ledger = JobLedger(tmp_path / "ledger.sqlite")
    work = tmp_path / "work"
    ledger.add("same", "design", "case_a", {"budget": 6}, work)
    assert ledger.claim("same")["attempts"] == 1
    assert ledger.claim("same") is None
    ledger.add("same", "design", "case_a", {"budget": 6}, work)
    with pytest.raises(ValueError, match="Frozen task changed"):
        ledger.add("same", "design", "case_a", {"budget": 7}, work)
    path = tmp_path / "result.json"
    write_json(path, {"status": "accepted"})
    ledger.finish("same", path)
    assert ledger.claim("same") is None


def test_generation_requires_current_protocol_and_code_preflight(tmp_path, monkeypatch):
    campaign = campaign_fixture(tmp_path)
    monkeypatch.setattr("agenticgen.campaign.code_fingerprint", lambda: "current-code")
    with pytest.raises(RuntimeError, match="current passing real-model preflight"):
        campaign.require_preflight()
    path = campaign.root / "preflight/result.json"
    receipt = {
        "status": "passed",
        "code_sha256": "old-code",
        "protocol_sha256": object_hash(campaign.config.model_dump(mode="json")),
    }
    write_json(path, receipt)
    with pytest.raises(RuntimeError, match="current passing real-model preflight"):
        campaign.require_preflight()
    write_json(path, {**receipt, "code_sha256": "current-code"})
    campaign.require_preflight()
    write_json(path, {**receipt, "code_sha256": "current-code", "protocol_sha256": "old-protocol"})
    with pytest.raises(RuntimeError, match="current passing real-model preflight"):
        campaign._generation({"payload": {"not_applicable": True}}, tmp_path / "worker")


def test_explicit_generation_phase_cannot_bypass_preflight(tmp_path):
    campaign = campaign_fixture(tmp_path)
    calls = []
    campaign.prepare = lambda: calls.append("prepare")
    campaign.enqueue_generation = lambda: calls.append("enqueue")
    campaign.phase = lambda *args: calls.append("run")

    def fail_preflight():
        calls.append("preflight")
        raise RuntimeError("Checkpoint preflight did not pass")

    campaign.preflight = fail_preflight
    with pytest.raises(RuntimeError, match="Checkpoint preflight did not pass"):
        campaign.run("generation")
    assert calls == ["prepare", "preflight"]


def generation_fixture(campaign, folder, *, affinity_ok=True):
    particles = [{"pose_id": f"p{i}", "status": "decode_failed"} for i in range(4)]
    write_json(
        folder / "poses.json",
        {
            "schema_version": "agenticgen.flowr-particles.v1",
            "sample_count": 4,
            "particles": particles,
        },
    )
    write_json(
        folder / "affinity.json",
        {
            "status": "completed" if affinity_ok else "failed",
            "poses": [
                {"pose_id": f"p{i}", "status": "unavailable", "pic50": None} for i in range(4)
            ],
        },
    )
    return {"status": "completed", "output_dir": str(folder), "attempted": 4}


def test_generation_reexecution_supersedes_old_completed_evaluation(tmp_path):
    campaign = campaign_fixture(tmp_path)
    generation_id = "generation:case_a:arm:batch_000"
    commit_job(
        campaign,
        generation_id,
        "generation",
        {},
        generation_fixture(campaign, tmp_path / "generation_v1"),
        tmp_path / "parent",
    )
    campaign.enqueue_evaluations()
    first = campaign.ledger.list(stage="evaluation")[0]
    campaign.ledger.claim(first["task_id"])
    path = tmp_path / "old_evaluation/result.json"
    write_json(path, {"status": "completed", "attempted": 4})
    campaign.ledger.finish(first["task_id"], path)
    campaign.ledger.requeue(generation_id, "Test model-affecting code repair")
    campaign.ledger.claim(generation_id)
    replacement = tmp_path / "parent_v2/result.json"
    write_json(replacement, generation_fixture(campaign, tmp_path / "generation_v2"))
    campaign.ledger.finish(generation_id, replacement)
    campaign.enqueue_evaluations()
    children = campaign.ledger.list(stage="evaluation")
    assert len(children) == 2
    old = campaign.ledger.get(first["task_id"])
    assert old["status"] == "superseded"
    current = next(job for job in children if job["status"] == "pending")
    assert current["task_id"] != first["task_id"]
    assert current["payload"]["generation_result_path"] == str(replacement)
    assert (
        current["payload"]["generation_result_sha256"]
        != first["payload"]["generation_result_sha256"]
    )
    assert path.is_file()  # A repaired run preserves the earlier result.


def test_evaluation_consumes_successful_affinity_recovery_instead_of_stale_parent(
    tmp_path, monkeypatch
):
    campaign = campaign_fixture(tmp_path)
    campaign._source_binding = lambda case_id: original_binding()
    campaign._components = SimpleNamespace(geometry=SimpleNamespace())
    parent_id = "generation:case_a:arm:batch_000"
    commit_job(
        campaign,
        parent_id,
        "generation",
        {},
        generation_fixture(campaign, tmp_path / "generation", affinity_ok=False),
        tmp_path / "parent",
    )
    campaign.enqueue_evaluations()
    assert campaign.ledger.list(stage="evaluation") == []
    recovery = campaign.ledger.list(stage="affinity")[0]
    campaign.ledger.claim(recovery["task_id"])
    scores = [{"pose_id": f"p{i}", "status": "ok", "pic50": 7.0 + i} for i in range(4)]
    path = tmp_path / "recovered/result.json"
    write_json(path, {"status": "completed", "poses": scores})
    campaign.ledger.finish(recovery["task_id"], path)
    campaign.enqueue_evaluations()
    evaluation = campaign.ledger.list(stage="evaluation")[0]
    received = []

    def evaluate(*args, affinity, **kwargs):
        received.extend(affinity)
        return [{"pose_id": row["pose_id"]} for row in affinity]

    monkeypatch.setattr("agenticgen.campaign.evaluate_particles", evaluate)
    result = campaign._evaluation(evaluation, tmp_path / "evaluation")
    assert result["attempted"] == 4
    assert received == scores
    assert evaluation["payload"]["affinity_result_path"] == str(path)


def test_affinity_singleton_native_priors_are_seed_order_and_batch_invariant(tmp_path, monkeypatch):
    """A stochastic native double detects prior seeding before collate, not just before head."""
    torch = pytest.importorskip("torch")
    from agenticgen.campaign_affinity import score_pose_files

    root = tmp_path / "flowr"
    root.mkdir()
    native = ModuleType("flowr.predict.predict_from_pdb")
    native.__file__ = str(root / "flowr/predict/predict_from_pdb.py")
    sizes, loaded_calls = [], []

    def get_args():
        def arg(name):
            return sys.argv[sys.argv.index(name) + 1]

        assert int(arg("--seed")) == 42
        assert float(arg("--coord_noise_scale")) == 0
        return SimpleNamespace(ligand_file=arg("--ligand_file"), seed=42)

    def load_data(args, **kwargs):
        mol = Chem.SDMolSupplier(args.ligand_file, removeHs=True)[0]
        return SimpleNamespace(
            metadata={},
            coords=torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32).unsqueeze(
                0
            ),
        )

    def dataset(systems, *args):
        sizes.append(len(systems))
        assert len(systems) == 1
        return systems

    def dataloader(args, systems, interpolant):
        # Prior creation consumes all three RNGs, just as transforms/collation can.
        source = systems[0]
        shift = random.random() + np.random.random()
        prior = {"coords": torch.randn_like(source.coords) + shift}
        posterior = {"coords": source.coords.clone()}
        for data in (prior, posterior):
            data.update(
                mask=torch.ones(source.coords.shape[:2], dtype=torch.long),
                fragment_mask=torch.zeros(source.coords.shape[:2], dtype=torch.bool),
                interactions=torch.zeros(1),
                complex=[source],
            )
        return [(prior, posterior, None, None)]

    class NativeModel:
        inpainting_mode = True
        graph_inpainting = True
        _inpaint_self_condition = True
        self_condition = False

        def __init__(self):
            self.builder = SimpleNamespace(
                extract_ligand_from_complex=lambda data: dict(data),
                extract_pocket_from_complex=lambda data: dict(data),
            )

        def to(self, device):
            return self

        def eval(self):
            return self

        def _predict_affinity(self, prior, data, pocket, times):
            assert len(data["coords"]) == 1
            assert self.self_condition and not self.graph_inpainting
            assert not self.inpainting_mode and not self._inpaint_self_condition
            assert float(times[0][0]) == pytest.approx(0.9999)
            # Depend on native SC prior and fresh randomness; task-seed or order
            # leakage would change this score even though ligand noise is zero.
            score = prior["coords"].sum() + data["coords"].sum() + torch.rand(())
            score += random.random() + np.random.random()
            return {"affinity": {"pic50": score.reshape(1)}}

    model = NativeModel()
    loaded = (model, {"remove_hs": True, "remove_aromaticity": True}, None, None, None, None)
    native.get_args = get_args
    native.load_data_from_pdb = load_data
    native.get_dataset = dataset
    native.get_dataloader = dataloader
    native.load_util = lambda *args: (None, None)
    native.load_model = lambda args: loaded_calls.append(args) or loaded
    device = ModuleType("flowr.util.device")
    device.resolve_device = lambda args: "cpu"
    device.dict_to_device = lambda data, target: data
    monkeypatch.setitem(sys.modules, "flowr.predict.predict_from_pdb", native)
    monkeypatch.setitem(sys.modules, "flowr.util.device", device)
    before_path = list(sys.path)
    monkeypatch.setattr(sys, "path", before_path)
    poses = []
    for index, shift in enumerate((0.0, 3.0)):
        mol = Chem.MolFromSmiles("CCO")
        conf = Chem.Conformer(3)
        conf.Set3D(True)
        for slot in range(3):
            conf.SetAtomPosition(slot, (shift + slot, float(slot % 2), 0.0))
        mol.AddConformer(conf)
        path = tmp_path / f"pose_{index}.sdf"
        with Chem.SDWriter(str(path)) as writer:
            writer.write(mol)
        poses.append({"pose_id": f"pose_{index}", "particle_index": index, "sdf_path": str(path)})
    poses.append({**poses[0], "pose_id": "same_pose_again", "particle_index": 2})
    task = {
        "flowr_root": str(root),
        "pocket_pdb": str(tmp_path / "pocket.pdb"),
        "checkpoint": str(tmp_path / "model.ckpt"),
        "output_dir": str(tmp_path / "scores"),
        "seed": 123,
    }
    torch.manual_seed(993)
    random.seed(892)
    np.random.seed(371)
    torch_before = torch.random.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    forward = score_pose_files(task, poses, loaded=loaded)
    backward = score_pose_files({**task, "seed": 999}, list(reversed(poses)), loaded=loaded)
    singleton = score_pose_files({**task, "seed": 2}, poses[:1], loaded=loaded)
    scores = {row["pose_id"]: row["pic50"] for row in forward}
    assert all(row["status"] == "ok" for row in forward + backward + singleton)
    assert {row["pose_id"]: row["pic50"] for row in backward} == scores
    assert singleton[0]["pic50"] == scores["pose_0"] == scores["same_pose_again"]
    assert sizes == [1] * 7
    assert loaded_calls == []
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert model.inpainting_mode and model.graph_inpainting
    assert model._inpaint_self_condition and not model.self_condition


@pytest.mark.parametrize("changed", [False, True])
def test_soft_guidance_lineage_measures_contacts_without_claiming_fixed_identity(tmp_path, changed):
    from agenticgen.campaign_evaluation import interaction_displacement, lineage_contacts
    from agenticgen.selection import contact_retention

    def pose(smiles, path):
        mol = Chem.MolFromSmiles(smiles)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for index in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(index, (float(index), 0.0, 0.0))
        mol.AddConformer(conf)
        with Chem.SDWriter(str(path)) as writer:
            writer.SetForceV3000(True)
            writer.write(mol)

    source, product = tmp_path / "source.sdf", tmp_path / "generated.sdf"
    pose("[CH3:1][OH:2]", source)
    pose("[CH3:1001][NH2:1002]" if changed else "[CH3:1001][OH:1002]", product)
    binding = original_binding().model_copy(
        update={
            "interactions": [Interaction(kind="HydrogenBond", residue="ASN118", atom_maps=[1002])]
        }
    )
    original = original_binding().model_copy(
        update={"interactions": [Interaction(kind="HydrogenBond", residue="ASN118", atom_maps=[2])]}
    )
    particle = {
        "slot_lineage": [
            {
                "reference_atom_map": reference,
                "generated_atom_map": generated,
                "local_graph_matches": True,
                "hard_atom_condition": False,
            }
            for reference, generated in ((1, 1001), (2, 1002))
        ]
    }
    aligned, mapping, receipt = lineage_contacts(source, product, particle, binding)
    retention = contact_retention(original, aligned)
    assert retention["fraction"] == (0 if changed else 1)
    assert receipt["lineage_is_a_hard_identity_constraint"] is False
    assert receipt["actual_identity_changed_original_maps"] == ([2] if changed else [])
    displacement = interaction_displacement(source, product, [2], mapping)
    assert displacement["rmsd_angstrom"] == 0
    assert displacement["missing_original_interaction_maps"] == []
    assert displacement["changed_original_interaction_identity_maps"] == ([2] if changed else [])
