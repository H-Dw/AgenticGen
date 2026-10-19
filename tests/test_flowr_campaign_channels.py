"""Real CPU tensor channel tests with a tiny native-interface model double."""

from types import SimpleNamespace

import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")

from agenticgen.region_conditions import RegionConditioner  # noqa: E402
from agenticgen.workers.flowr_campaign import (  # noqa: E402
    _load_model_cached,
    capture_native_probe_state,
    export_particles,
    native_condition_channels,
)


def values():
    return {
        "coords": torch.tensor(
            [[[90.0, 91.0, 92.0], [100.0, 101.0, 102.0], [110.0, 111.0, 112.0]]]
        ),
        "atomics": torch.eye(3).unsqueeze(0),
        "charges": torch.eye(3).unsqueeze(0),
        "bonds": torch.eye(3).reshape(1, 3, 3, 1).repeat(1, 1, 1, 3),
        "mask": torch.ones(1, 3, dtype=torch.long),
    }


def policy(mode):
    return {
        "mode": mode,
        "fixed_atom_input_slots": [0, 1],
        "fixed_bond_pairs": [[0, 0], [0, 1], [1, 1], [1, 2]],
        "fixed_coordinate_input_slots": [0] if mode == "fragment_graph_pose" else [],
    }


def noise():
    result = values()
    result.update(
        coords=torch.arange(9.0).reshape(1, 3, 3),
        atomics=torch.zeros(1, 3, 3),
        charges=torch.zeros(1, 3, 3),
        bonds=torch.zeros(1, 3, 3, 3),
    )
    return result


@pytest.mark.parametrize("mode", ["graph", "fragment_graph"])
def test_graph_only_prior_and_self_condition_never_copy_reference_coordinates(mode):
    condition = RegionConditioner(policy(mode), values())
    prior = noise()
    conditioned = condition.apply(prior)
    assert torch.equal(conditioned["coords"], prior["coords"])
    assert torch.equal(conditioned["atomics"][:, :2], values()["atomics"][:, :2])
    assert torch.equal(conditioned["atomics"][:, 2], prior["atomics"][:, 2])
    sc = condition.apply(
        conditioned, feature_keys=["coords", "atomics", "bonds"], overwrite_with_zeros=True
    )
    assert torch.equal(sc["coords"], prior["coords"])
    assert not bool((sc["coords"] == values()["coords"]).all())
    assert not bool(sc["charges"][:, :2].any())
    assert torch.equal(sc["charges"][:, 2], prior["charges"][:, 2])


def test_native_sc_key_list_cannot_omit_fixed_charge_or_hybridization():
    reference, predicted = values(), noise()
    reference["hybridization"] = torch.eye(3).unsqueeze(0)
    predicted["hybridization"] = torch.zeros(1, 3, 3)
    condition = RegionConditioner(policy("fragment_graph"), reference)
    result = condition.apply(predicted, feature_keys=["coords", "atomics", "bonds"])
    for key in ("charges", "hybridization"):
        assert torch.equal(result[key][:, :2], reference[key][:, :2])
        assert torch.equal(result[key][:, 2], predicted[key][:, 2])


def test_pose_channel_copies_only_i_while_edited_coordinates_are_free():
    condition = RegionConditioner(policy("fragment_graph_pose"), values())
    prior = noise()
    result = condition.apply(prior)
    assert torch.equal(result["coords"][:, 0], values()["coords"][:, 0])
    assert torch.equal(result["coords"][:, 1:], prior["coords"][:, 1:])
    # final graph clamp excludes coords in native feature_keys; I still stays fixed.
    final = condition.apply(prior, feature_keys=["atomics", "bonds", "charges"])
    assert torch.equal(final["coords"][:, 0], values()["coords"][:, 0])
    assert torch.equal(final["coords"][:, 1], prior["coords"][:, 1])


def test_independent_clocks_and_bond_masks():
    condition = RegionConditioner(policy("fragment_graph_pose"), values())
    continuous, discrete = condition.times(
        [torch.tensor([0.4]), torch.tensor([0.4])], values()["mask"]
    )
    assert torch.allclose(continuous.flatten(), torch.tensor([1.0, 0.4, 0.4]))
    assert torch.allclose(discrete.flatten(), torch.tensor([1.0, 1.0, 0.4]))
    assert bool(condition.bond_mask[0, 1, 2])  # optimized boundary in both directions.
    assert bool(condition.bond_mask[0, 2, 1])
    assert not bool(condition.bond_mask[0, 0, 2])


def test_runtime_embedding_channels_restore_after_failure():
    def original(*args, **kwargs):
        return kwargs

    model = SimpleNamespace(
        graph_inpainting=False,
        inpainting_mode=False,
        inpainting_mode_inf=None,
        use_is_fixed_embed=True,
        gen=SimpleNamespace(forward=original),
        builder=SimpleNamespace(inpaint_graph=original),
        _get_times=original,
    )
    condition = RegionConditioner(policy("fragment_graph_pose"), values())
    with pytest.raises(RuntimeError, match="sampling failure"):
        with native_condition_channels(model, condition):
            embeddings = model.gen.forward()
            assert embeddings["is_fixed_node"].tolist() == [[1, 1, 0]]
            assert embeddings["is_fixed_position"].tolist() == [[1, 0, 0]]
            assert model.graph_inpainting and not model.inpainting_mode
            raise RuntimeError("sampling failure")
    assert not model.graph_inpainting and model.inpainting_mode_inf is None
    assert model.gen.forward is original and model.builder.inpaint_graph is original


def test_particle_export_keeps_failures_ids_and_large_maps_without_selection(tmp_path):
    mol = Chem.MolFromSmiles("CO")
    conf = Chem.Conformer(2)
    conf.Set3D(True)
    conf.SetAtomPosition(0, (1.0, 2.0, 3.0))
    conf.SetAtomPosition(1, (2.0, 2.0, 3.0))
    mol.AddConformer(conf)
    task = {
        "sample_count": 2,
        "mode": "fragment_graph",
        "seed": 4,
        "output_dir": str(tmp_path),
        "particle_ids": ["physical_000", "physical_001"],
    }
    raw = {"mask": torch.ones(2, 2, dtype=torch.long), "atomics": torch.ones(2, 2, 3)}
    result = export_particles(
        task,
        {"atom_maps": [7, 501], "product_mapped_smiles": "[CH3:7][OH:501]"},
        raw,
        [mol, None],
        {"fixed_atom_input_slots": [0]},
        SimpleNamespace(token_idx_map={}),
        torch,
    )
    assert len(result["particles"]) == 2
    good, bad = result["particles"]
    assert good["pose_id"] == "physical_000" and good["status"] == "decoded"
    loaded = Chem.SDMolSupplier(good["sdf_path"])[0]
    assert [a.GetAtomMapNum() for a in loaded.GetAtoms()] == [7, 1502]
    assert "accepted_for_evaluation" not in good
    assert bad["pose_id"] == "physical_001" and bad["status"] == "decode_failed"
    assert good["tensor_slot_to_reference_atom_map"] == {0: 7, 1: 501}
    assert bad["tensor_slot_to_reference_atom_map"] == {0: 7, 1: 501}
    assert good["slot_lineage"][1] == {
        "input_slot": 1,
        "reference_atom_map": 501,
        "generated_atom_map": 1502,
        "element_matches": True,
        "charge_matches": True,
        "local_graph_matches": True,
        "hard_atom_condition": False,
    }


def test_soft_particle_lineage_records_chemistry_changes_without_claiming_identity(tmp_path):
    mol = Chem.MolFromSmiles("CN")
    mol.AddConformer(Chem.Conformer(2))
    result = export_particles(
        {"sample_count": 1, "mode": "gradient", "seed": 2, "output_dir": str(tmp_path)},
        {"atom_maps": [7, 501], "product_mapped_smiles": "[CH3:7][OH:501]"},
        {"mask": torch.ones(1, 2, dtype=torch.long), "atomics": torch.ones(1, 2, 3)},
        [mol],
        {"fixed_atom_input_slots": []},
        SimpleNamespace(token_idx_map={}),
        torch,
    )["particles"][0]
    assert result["tensor_slot_to_atom_map"] == {0: 1501, 1: 1502}
    assert not result["slot_lineage"][1]["element_matches"]
    assert result["slot_lineage"][1]["reference_atom_map"] == 501
    assert not any(row["hard_atom_condition"] for row in result["slot_lineage"])


def test_probe_capture_preserves_sc_flags_omits_duplicate_training_and_does_not_draw_rng(tmp_path):
    calls = []

    def forward(batch, pocket, times, **kwargs):
        calls.append(kwargs)
        return batch

    model = SimpleNamespace(forward=forward)
    batch = noise()
    sc = {"coords": batch["coords"].clone(), "_sc_fired": True}
    pocket = {"coords": torch.ones(1, 2, 3), "complex": [SimpleNamespace(com=torch.ones(1, 3))]}
    state = torch.random.get_rng_state().clone()
    destination = tmp_path / "guidance_probe_state.pt"
    with capture_native_probe_state(model, destination, torch) as capture:
        times = [torch.tensor([0.4])] * 3
        model.forward(
            batch, pocket, times, cond_batch=sc, training=False, pocket_equis=torch.ones(1)
        )
        assert capture["cond_batch"]["_sc_fired"] is True
        assert "training" not in capture["forward_kwargs"]
        batch["coords"].add_(100)
        assert not torch.equal(batch["coords"], capture["batch"]["coords"])
    saved = torch.load(destination, weights_only=True)
    assert saved["cond_batch"]["_sc_fired"] is True
    assert "training" not in saved["forward_kwargs"]
    assert "complex" not in saved["pocket_batch"]
    assert torch.equal(saved["pocket_com_world"][0], torch.ones(1, 3))
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.forward is forward and calls[0]["training"] is False


def test_cached_model_restores_initial_mutable_configuration(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fake-checkpoint")
    args = SimpleNamespace(save_dir="execution_001")
    model = SimpleNamespace(
        hparams={"sc_charges": True, "nested": {"value": 3}},
        builder=SimpleNamespace(save_dir="initial", ligand_valence_repair=False),
        graph_inpainting=False,
        self_condition=True,
        inpainting_mode=False,
        inpainting_mode_inf=None,
    )
    loaded = (model, {"remove_aromaticity": True, "nested": {"value": 5}}, *([None] * 6))
    loads = []
    native = SimpleNamespace(load_model=lambda config: loads.append(config) or loaded)
    task = {"checkpoint": str(checkpoint), "flowr_root": str(tmp_path)}
    cache = {}
    _, first_hit = _load_model_cached(native, args, task, cache)
    model.hparams["nested"]["value"] = 50
    loaded[1]["nested"]["value"] = 100
    model.graph_inpainting, model.self_condition = True, False
    model.builder.ligand_valence_repair = True
    second, second_hit = _load_model_cached(
        native, SimpleNamespace(save_dir="execution_002"), task, cache
    )
    assert not first_hit and second_hit and len(loads) == 1
    assert second is loaded
    assert loaded[1]["nested"]["value"] == 5
    assert model.hparams["nested"]["value"] == 3
    assert not model.graph_inpainting and model.self_condition
    assert not model.builder.ligand_valence_repair
    assert (
        loaded[1]["save_dir"]
        == model.hparams["save_dir"]
        == model.builder.save_dir
        == "execution_002"
    )
