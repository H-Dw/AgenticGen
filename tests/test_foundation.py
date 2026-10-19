from dataclasses import replace

import pytest

from agenticgen.base_config import FoundationConfig, load_foundation
from agenticgen.base_factory import build_foundation
from agenticgen.base_runner import foundation_run
from agenticgen.chemistry import graph_hash, mapped_smiles
from agenticgen.contracts import BindingResult, Interaction, Observation, RetroResult
from agenticgen.corpus import build_corpus
from agenticgen.dataset import load_dataset
from agenticgen.enhancements import EvidenceInjector
from agenticgen.io import read_json, write_json
from agenticgen.selection import contact_retention, select_candidates


def configuration(repo, tmp_path, **updates):
    value = read_json(repo / "examples/syncraft_foundation/standard.inference.json")
    value["dataset"] = str(repo / "examples/syncraft_foundation/ck2_one.json")
    value["syncraft"]["root"] = str(repo / "external/SynCraft-Core")
    value["llm"].update(
        backend="replay", replay_file=str(repo / "examples/syncraft_foundation/native_replay.json")
    )
    value.update(updates)
    path = tmp_path / "config.json"
    write_json(path, value)
    return load_foundation(path)


@pytest.mark.integration
def test_native_pass_k_uses_upstream_prompt_and_cache_before_selection(
    repo, tmp_path, syncraft_root
):
    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_foundation(config, dataset)
    output = tmp_path / "run"
    result = foundation_run(config, dataset, components, output)
    assert result["proposal_count"] == 5 and result["accepted_edit_count"] == 5
    assert result["execution_kind"] == "engineering_replay"
    assert result["selected_case_count"] == 0  # Inference-only never claims a winner.
    folder = output / "cases" / dataset.cases[0].case_id
    request = read_json(folder / "calls/proposal_00.request.json")
    assert "[EXAMPLES OF HIGH-QUALITY SOLUTION]" in request["messages"][1]["content"]
    assert "[YOUR TASK]" in request["messages"][1]["content"]
    decisions = read_json(folder / "decision.json")
    assert len(decisions["proposals"]) == 5
    assert all("properties" not in p and "graph" not in p for p in decisions["proposals"])
    saved = read_json(folder / "result.json")
    manifest = read_json(output / "manifest.json")
    assert manifest["executed_case_ids"] == [c.case_id for c in dataset.cases]
    assert saved["candidates"][1]["cache_hits"] == {"retro": True, "binding": True}
    assert saved["source_binding"]["status"] == "not_run"
    assert saved["route_rescue"] is None


def test_unknown_case_subset_fails_before_output_creation(repo, tmp_path):
    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    output = tmp_path / "invalid_subset"
    with pytest.raises(ValueError, match="unknown case"):
        foundation_run(config, dataset, None, output, case_ids=["absent"])
    assert not output.exists()


@pytest.mark.integration
@pytest.mark.parametrize("invalid", [False, True])
def test_native_fragment_selection_and_unreadable_outputs_reach_runner(
    repo, tmp_path, syncraft_root, invalid
):
    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    case_id = dataset.cases[0].case_id
    commands = (
        [["DEL_ATOM", i] for i in range(11, 16)]
        if invalid
        else [["DEL_ATOM", 20], ["DEL_ATOM", 21]]
    ) + [["STOP"]]
    import json

    fixture = tmp_path / "fragment_replay.json"
    write_json(
        fixture,
        {
            "kind": "engineering_native_replay",
            "responses": {
                case_id: [
                    "[REASONING]\nSaved graph-edit fixture.\n[EDITS]\n```json\n"
                    + json.dumps(commands)
                    + "\n```"
                ]
                * 5
            },
        },
    )
    config = replace(
        config,
        value=config.value.model_copy(
            update={"llm": config.value.llm.model_copy(update={"replay_file": str(fixture)})}
        ),
    )
    output = tmp_path / "native_run"
    result = foundation_run(config, dataset, build_foundation(config, dataset), output)
    saved = read_json(output / "cases" / case_id / "result.json")
    assert result["graph_reconstruction"] == "syncraft_native"
    if invalid:
        assert result["accepted_edit_count"] == 0
        assert result["product_loading_failure_count"] == 5
        assert all("properties" not in c for c in saved["candidates"])
        assert all(c["graph"]["upstream_smiles"] for c in saved["candidates"])
    else:
        assert result["accepted_edit_count"] == result["fragment_pruned_proposal_count"] == 5
        assert result["unique_accepted_product_count"] == 1
        assert saved["candidates"][0]["graph"]["discarded_atom_maps"] == [22]
        assert saved["candidates"][0]["proposal"]["edits"] == commands


@pytest.mark.integration
def test_historical_summary_keeps_original_editor_label(repo, tmp_path, syncraft_root):
    from agenticgen.base_reporting import rebuild_foundation_summary

    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    output = tmp_path / "historical"
    foundation_run(config, dataset, build_foundation(config, dataset), output)
    manifest = read_json(output / "manifest.json")
    manifest.pop("graph_reconstruction")
    manifest["effective_config"]["editor"] = "strict"
    write_json(output / "manifest.json", manifest)
    result = rebuild_foundation_summary(output)
    assert result["graph_reconstruction"] == "strict"
    assert read_json(output / "manifest.json") == manifest


def candidate(index, *, sim=0.7, sa=3.0, route="route_found", verified=False, vina=-7):
    return {
        "proposal_index": index,
        "status": "accepted",
        "selection_similarity": sim,
        "properties": {"status": "ok", "value": sa},
        "binding": {"status": "ok", "score_kcal_mol": vina},
        "retrosynthesis": {"status": route, "route_verified": verified},
    }


def test_paper_selection_is_route_then_similarity_not_sa():
    from agenticgen.base_config import SelectionConfig

    binding = BindingResult(status="not_run", protocol_id="off", graph_smiles="CC")
    settings = SelectionConfig()
    candidates = [
        candidate(0, sim=0.9, sa=1.2, route="no_route_within_budget"),
        candidate(1, sim=0.7, sa=3.5),
        candidate(2, sim=0.6, sa=2),
        candidate(3, sim=0.5, verified=True),
    ]
    chosen = select_candidates(candidates, settings, binding)
    assert chosen["selected_proposal_index"] == 1
    assert chosen["eligible_proposal_indices"] == [1, 2]
    strict = settings.model_copy(update={"require_verified_route": True})
    assert select_candidates(candidates, strict, binding)["selected_proposal_index"] is None


def test_skill_selection_rejects_docking_loss_and_ranks_low_sa():
    from agenticgen.base_config import SelectionConfig

    binding = BindingResult(
        status="ok", protocol_id="fixture", graph_smiles="CC", score_kcal_mol=-8
    )
    settings = SelectionConfig(
        policy="skill_sa_binding", similarity_threshold=0.2, similarity_bits=2048
    )
    records = [
        candidate(0, sa=1, vina=-6.9),
        candidate(1, sa=2.5, vina=-7),
        candidate(2, sa=2.2, vina=-7.5),
    ]
    assert select_candidates(records, settings, binding)["selected_proposal_index"] == 2
    records[2]["properties"] = {"status": "unavailable", "value": None}
    assert select_candidates(records, settings, binding)["selected_proposal_index"] == 1


@pytest.mark.integration
def test_route_policy_uses_fixed_candidate_set_and_original_denominator(
    repo, tmp_path, syncraft_root
):
    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_foundation(config, dataset)

    class Evaluator:
        calls = 0

        def evaluate(self, smiles, output):
            self.calls += 1
            # Every proposal must have been committed before the original route search.
            decision = tmp_path / "route_run/cases" / dataset.cases[0].case_id / "decision.json"
            assert len(read_json(decision)["proposals"]) == 5
            return RetroResult(
                status="no_route_within_budget" if self.calls == 1 else "route_found",
                protocol_id="fixture",
            )

    # Constructed components are injected; no real search or fabricated production backend.
    from agenticgen.base_config import SelectionConfig
    from agenticgen.config import RetroConfig

    configured = config.value.model_copy(
        update={
            "selection": SelectionConfig(),
            "retro": RetroConfig(
                backend="simpretro_cli",
                root="fixture",
                python="fixture",
                inventory="fixture",
                templates="fixture",
            ),
        }
    )
    evaluator = Evaluator()
    summary = foundation_run(
        replace(config, value=configured),
        dataset,
        replace(components, retro=evaluator),
        tmp_path / "route_run",
    )
    assert summary["physical_retro_queries"] == evaluator.calls == 2
    assert summary["route_rescue_case_count"] == 1
    assert summary["route_verified_count"] == 0


@pytest.mark.integration
def test_interaction_prompt_and_contact_identity_use_source_maps(repo, tmp_path, syncraft_root):
    config = configuration(
        repo,
        tmp_path,
        mode="interaction_aware",
        geometry={
            "backend": "syncraft_worker",
            "python": "fixture",
            "receptor_pdbqt": "fixture",
            "receptor_pdb": "fixture",
            "vina_config": "fixture",
        },
    )
    # Test prompt in isolation without claiming a configured docking runtime exists.
    standard = configuration(repo, tmp_path)
    dataset = load_dataset(standard.resolve(standard.value.dataset))
    components = build_foundation(standard, dataset)
    components.agent.mode = config.value.mode
    case = dataset.cases[0]
    observation = Observation(
        case_id=case.case_id,
        target="CK2",
        source_smiles=case.smiles,
        mapped_smiles=mapped_smiles(case.smiles),
        graph_sha256=graph_hash(case.smiles),
        terminal_metrics={},
        constraints=config.value.constraints,
    )
    original = BindingResult(
        status="ok",
        protocol_id="fixture",
        graph_smiles=case.smiles,
        score_kcal_mol=-8,
        prompt="[CRITICAL BIOLOGICAL CONSTRAINTS; PRESERVE]\nAtom [2] contacts LYS68.A",
        interactions=[Interaction(kind="HydrogenBond", residue="LYS68.A", atom_maps=[2])],
    )
    messages = components.agent.messages(observation, original)
    assert "Atom [2] contacts LYS68.A" in messages[1]["content"]
    changed = original.model_copy(
        update={
            "interactions": [Interaction(kind="HydrogenBond", residue="LYS68.A", atom_maps=[500])]
        }
    )
    assert contact_retention(original, changed)["fraction"] == 0


def test_enhancements_are_opt_in_bound_to_graph_and_share_baseline_config(repo, tmp_path):
    from agenticgen.base_config import EnhancementConfig

    dataset = load_dataset(repo / "examples/syncraft_foundation/ck2_one.json")
    assert EvidenceInjector(EnhancementConfig(), tmp_path / "missing", dataset).packets == {}
    case = dataset.cases[0]
    path = tmp_path / "evidence.json"
    packet = {
        "case_id": case.case_id,
        "source_graph_sha256": graph_hash(case.smiles),
        "kind": "flow_history",
        "text": "Measured S_t changes; no chemical mechanism claim.",
        "evidence_ids": ["frame_20", "frame_40"],
    }
    write_json(path, {"schema_version": "agenticgen.enhancements.v1", "packets": [packet]})
    settings = EnhancementConfig(enabled=True, packets_file=str(path))
    assert (
        EvidenceInjector(settings, path, dataset).for_case(case.case_id)[0]["text"]
        == packet["text"]
    )
    packet["source_graph_sha256"] = "wrong"
    write_json(path, {"schema_version": "agenticgen.enhancements.v1", "packets": [packet]})
    with pytest.raises(ValueError, match="different source"):
        EvidenceInjector(settings, path, dataset)


@pytest.mark.integration
def test_corpus_builder_executes_upstream_mcs_and_checks_target(syncraft_root, tmp_path):
    pairs = tmp_path / "pairs.json"
    write_json(
        pairs,
        [
            {
                "source": "CCCO",
                "target": "CCCN",
                "reasoning": "Engineering graph transformation fixture.",
                "evidence_ids": ["fixture_not_synthesis_label"],
            }
        ],
    )
    output = tmp_path / "corpus.json"
    result = build_corpus(syncraft_root, pairs, output)
    assert result["accepted"] == 1
    assert read_json(output)[0]["target"] == "CCCN"


def test_full_profiles_require_evaluators_and_no_silent_fallback(repo):
    value = read_json(repo / "examples/syncraft_foundation/standard.inference.json")
    value["selection"] = {"policy": "paper_route_similarity"}
    with pytest.raises(ValueError, match="real retrosynthesis"):
        FoundationConfig.model_validate(value)
    value["selection"] = {"policy": "proposal_only"}
    value["mode"] = "interaction_aware"
    with pytest.raises(ValueError, match="geometry"):
        FoundationConfig.model_validate(value)


@pytest.mark.integration
def test_dual_freezes_both_modes_and_deduplicates_union(repo, tmp_path, syncraft_root):
    from agenticgen.base_config import GeometryConfig, SelectionConfig

    config = configuration(repo, tmp_path)
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_foundation(config, dataset)
    settings = config.value.model_copy(
        update={
            "mode": "dual",
            "pass_k": 2,
            "selection": SelectionConfig(policy="skill_sa_binding", similarity_threshold=0.2),
            "geometry": GeometryConfig(
                backend="syncraft_worker",
                python="fixture",
                receptor_pdbqt="fixture",
                receptor_pdb="fixture",
                vina_config="fixture",
            ),
        }
    )

    class Geometry:
        calls = 0

        def evaluate(self, smiles, output):
            self.calls += 1
            if self.calls > 1:
                saved = read_json(
                    tmp_path / "dual/cases" / dataset.cases[0].case_id / "decision.json"
                )
                assert len(saved["proposals"]) == 4
            return BindingResult(
                status="ok",
                protocol_id="fixture",
                graph_smiles=smiles,
                score_kcal_mol=-8,
                prompt="UNIQUE_PLIP_MARKER Atom [2] contacts LYS68.A",
            )

    geometry = Geometry()
    summary = foundation_run(
        replace(config, value=settings),
        dataset,
        replace(components, geometry=geometry),
        tmp_path / "dual",
    )
    assert summary["mode_proposal_counts"] == {"standard": 2, "interaction_aware": 2}
    assert summary["unique_accepted_product_count"] == 1
    assert geometry.calls == 2
    folder = tmp_path / "dual/cases" / dataset.cases[0].case_id
    standard = read_json(folder / "calls/proposal_00.request.json")
    interaction = read_json(folder / "calls/proposal_02.request.json")
    assert "UNIQUE_PLIP_MARKER" not in standard["messages"][1]["content"]
    assert "UNIQUE_PLIP_MARKER" in interaction["messages"][1]["content"]
    result = read_json(folder / "result.json")
    assert result["selection"]["eligible_proposal_indices"] == [0]
    assert result["per_mode_selection"]["interaction_aware"]["selected_proposal_index"] == 2
    assert result["route_validated_selection"]["selected_proposal_index"] is None
    assert result["verified_route_rescue"] is None


def test_paper_contact_metric_does_not_require_atom_identity():
    binding = BindingResult(
        status="ok",
        protocol_id="test",
        graph_smiles="CC",
        score_kcal_mol=-8,
        interactions=[Interaction(kind="HydrogenBond", residue="LYS68.A", atom_maps=[1])],
    )
    edited = binding.model_copy(
        update={
            "interactions": [Interaction(kind="HydrogenBond", residue="LYS68.A", atom_maps=[500])]
        }
    )
    result = contact_retention(binding, edited)
    assert result["fraction"] == 0
    assert result["paper_recall"] == result["paper_jaccard"] == 1
