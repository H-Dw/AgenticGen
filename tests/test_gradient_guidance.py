from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from agenticgen.gradient_guidance import (  # noqa: E402
    FlowrGuidanceController,
    GuidanceConfig,
    RegionalReward,
    RegionTargets,
    build_controller,
    coordinate_displacement,
    normalized_logit_update,
    probe_model_reward,
    targets_from_conditions,
)


def targets(dtype=torch.float64):
    return RegionTargets(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 0]),
        torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        torch.tensor([True, True, False]),
        torch.tensor([[False, True, True], [True, False, False], [True, False, False]]),
        torch.tensor([True, False, False]),
        torch.zeros((3, 3), dtype=dtype),
        (5, 7, 99),
    )


def prediction(batch=1, dtype=torch.float64):
    return {
        "atomics": torch.zeros((batch, 3, 3), dtype=dtype, requires_grad=True),
        "charges": torch.zeros((batch, 3, 2), dtype=dtype, requires_grad=True),
        "bonds": torch.zeros((batch, 3, 3, 2), dtype=dtype, requires_grad=True),
        "coords": torch.ones((batch, 3, 3), dtype=dtype, requires_grad=True),
        "mask": torch.ones((batch, 3), dtype=torch.bool),
    }


@pytest.mark.parametrize("family", ["ce", "brier"])
def test_balanced_graph_loss_counts_unique_pairs_and_keeps_support(family):
    out, tar = prediction(), targets()
    value, detail = RegionalReward(tar, GuidanceConfig(family=family))(out, out["coords"])
    assert set(detail) == {"atom", "charge", "bond_present", "bond_absent", "graph", "total"}
    if family == "ce":
        expected = (torch.log(torch.tensor(3.0)) + 3 * torch.log(torch.tensor(2.0))) / 4
        assert float(-value[0].detach()) == pytest.approx(float(expected), rel=1e-6)
    grad_a, grad_c, grad_b = torch.autograd.grad(
        value.sum(), (out["atomics"], out["charges"], out["bonds"])
    )
    assert torch.count_nonzero(grad_a[:, 2]) == 0
    assert torch.count_nonzero(grad_c[:, 2]) == 0
    assert torch.count_nonzero(grad_b[:, 1, 2]) == 0
    assert torch.count_nonzero(grad_b.diagonal(dim1=1, dim2=2)) == 0
    assert torch.equal(grad_b, grad_b.transpose(1, 2))
    assert grad_a[0, 0, 0] > 0
    assert grad_b[0, 0, 1, 1] > 0
    assert grad_b[0, 0, 2, 0] > 0


@pytest.mark.parametrize("family", ["ce", "brier"])
def test_graph_derivative_finite_difference_and_guidance_improves_target(family):
    out, tar = prediction(), targets()
    reward = RegionalReward(tar, GuidanceConfig(family=family))
    value, _ = reward(out, out["coords"])
    grads = torch.autograd.grad(value.sum(), (out["atomics"], out["charges"], out["bonds"]))
    by_key = dict(zip(("atomics", "charges", "bonds"), grads, strict=True))
    eps = 1e-5
    plus, minus = dict(out), dict(out)
    direction = grads[0] / grads[0].norm()
    plus["atomics"] = out["atomics"] + eps * direction
    minus["atomics"] = out["atomics"] - eps * direction
    numerical = (reward(plus, out["coords"])[0] - reward(minus, out["coords"])[0]) / (2 * eps)
    assert float(numerical.detach()) == pytest.approx(float(grads[0].norm()), rel=1e-6)
    probs, info = normalized_logit_update(out, by_key, tar, weight=1, dt=0.01)
    assert probs["atomics"][0, 0, 0] > 1 / 3
    assert torch.equal(probs["bonds"], probs["bonds"].transpose(1, 2))
    assert torch.allclose(probs["atomics"][:, 2], torch.full((1, 3), 1 / 3, dtype=torch.float64))
    assert info["atomics_kl"].item() > 0
    assert torch.allclose(probs["atomics"].sum(-1), torch.ones((1, 3), dtype=torch.float64))


def test_reward_batch_independence_and_pseudohuber_coordinate_sign():
    tar = targets()
    reward = RegionalReward(tar, GuidanceConfig(coordinates=True))
    out = prediction(2)
    value, _ = reward(out, out["coords"])
    grad = torch.autograd.grad(value.sum(), out["coords"])[0]
    assert bool((grad[:, 0] < 0).all())
    assert torch.count_nonzero(grad[:, 1:]) == 0
    single = prediction(1)
    single_value, _ = reward(single, single["coords"])
    single_grad = torch.autograd.grad(single_value.sum(), single["coords"])[0]
    assert torch.equal(grad[:1], single_grad)
    exact = torch.zeros_like(out["coords"], requires_grad=True)
    value, _ = reward(out, exact)
    assert torch.count_nonzero(torch.autograd.grad(value.sum(), exact)[0]) == 0


def test_displacement_uses_native_step_and_caps_atoms():
    grad = torch.tensor([[[1.0, 0, 0], [0, 2, 0], [99, 99, 99]]])
    native = torch.ones_like(grad) * 0.1
    mask = torch.tensor([[True, True, False]])
    delta, info = coordinate_displacement(grad, native, mask, weight=4, coord_scale=2)
    assert float(delta.norm(dim=-1).max() * 2) <= 0.02500001
    assert torch.count_nonzero(delta[:, 2]) == 0
    assert info["clipped_atom_fraction"].item() == 1
    assert info["native_step_rms_angstrom"].item() == pytest.approx(0.2)
    zero, _ = coordinate_displacement(torch.zeros_like(grad), native, mask, 4, 2)
    assert torch.count_nonzero(zero) == 0
    zero, _ = coordinate_displacement(grad, torch.zeros_like(native), mask, 4, 2)
    assert torch.count_nonzero(zero) == 0


class FakeIntegrator:
    def step(self, curr, predicted, prior, times, step_size):
        # A stochastic native step lets zero-weight tests verify exact RNG use.
        result = dict(predicted)
        result["coords"] = curr["coords"] + step_size * (
            predicted["coords"] - curr["coords"] + torch.randn_like(curr["coords"]) * 0.01
        )
        return result


class FakeFlowr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.coord_scale = 2.0
        self.graph_inpainting = False
        self.inpainting_mode = False
        self.integrator = FakeIntegrator()

    def forward(self, batch, pocket_batch, t, cond_batch=None, **kwargs):
        x = batch["coords"]
        signal = x.sum(-1, keepdim=True) * self.scalar
        pair = signal[:, :, None] + signal[:, None, :]
        return {
            "coords": x * self.scalar + 0.2,
            "atomics": torch.cat((signal, -signal, signal * 0.2), dim=-1),
            "charges": torch.cat((signal * 0.3, -signal * 0.3), dim=-1),
            "bonds": torch.cat((pair, -pair), dim=-1),
            "mask": batch["mask"],
        }


def fake_state():
    torch.manual_seed(7)
    batch = prediction()
    batch["coords"] = torch.randn((1, 3, 3), dtype=torch.float64)
    pocket = {"complex": [SimpleNamespace(com=torch.zeros(3))]}
    return batch, pocket


def native_one_step(model, batch, pocket, time=0.2):
    clock = [torch.tensor([time])] * 3
    with torch.no_grad():
        out = model(batch, pocket, clock)
        predicted = {
            key: torch.softmax(value, -1) if key in {"atomics", "bonds", "charges"} else value
            for key, value in out.items()
        }
        return model.integrator.step(batch, predicted, batch, clock, 0.01)


def test_zero_weight_has_bitwise_native_and_rng_parity():
    model, (batch, pocket) = FakeFlowr().eval(), fake_state()
    controller = FlowrGuidanceController(targets(), GuidanceConfig(weight=0))
    torch.manual_seed(123)
    expected = native_one_step(model, batch, pocket)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(123)
    with controller.install(model):
        actual = native_one_step(model, batch, pocket)
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert controller.trace == []
    assert "forward" not in model.__dict__
    assert "step" not in model.integrator.__dict__


def test_runtime_uses_live_model_gradient_and_restores_methods_and_parameters():
    model, (batch, pocket) = FakeFlowr().eval(), fake_state()
    controller = FlowrGuidanceController(targets(), GuidanceConfig(coordinates=True))
    original = model.scalar.requires_grad
    with controller.install(model):
        assert not model.scalar.requires_grad
        result = native_one_step(model, batch, pocket)
    assert model.scalar.requires_grad == original
    assert not result["coords"].requires_grad
    assert controller.trace[0]["coordinate_gradient_rms"][0] > 0
    assert controller.trace[0]["atomics_kl"][0] > 0
    assert controller.trace[0]["injected_max_atom_angstrom"][0] <= 0.02500001
    assert "forward" not in model.__dict__


def test_runtime_suffix_remains_native_and_window_has_exactly_80_steps():
    config = GuidanceConfig()
    assert sum(config.active(i / 100) for i in range(100)) == 80
    assert config.active(0.099999986)
    assert not config.active(0.9000001)
    model, (batch, pocket) = FakeFlowr().eval(), fake_state()
    controller = FlowrGuidanceController(targets(), config)
    torch.manual_seed(29)
    expected = native_one_step(model, batch, pocket, time=0.9)
    torch.manual_seed(29)
    with controller.install(model):
        actual = native_one_step(model, batch, pocket, time=0.9)
    assert torch.equal(expected["coords"], actual["coords"])
    assert controller.trace == []


def test_model_preflight_checks_vjp_and_all_logit_derivatives_and_preserves_rng():
    model, (batch, pocket) = FakeFlowr().eval(), fake_state()
    controller = FlowrGuidanceController(targets(), GuidanceConfig(coordinates=True))
    rng = torch.random.get_rng_state().clone()
    result = probe_model_reward(
        controller, model, batch, pocket, [torch.tensor([0.4])] * 3, epsilon=0.00001
    )
    assert result["passed"]
    assert set(result["derivatives"]) == {"current_coordinate_VJP", "atomics", "charges", "bonds"}
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.scalar.requires_grad


def test_runtime_rejects_hard_inpainting_and_restores_after_failure():
    model = FakeFlowr().eval()
    controller = FlowrGuidanceController(targets(), GuidanceConfig())
    model.graph_inpainting = True
    with pytest.raises(ValueError, match="hard inpainting"), controller.install(model):
        pass
    model.graph_inpainting = False
    with pytest.raises(RuntimeError, match="forced"), controller.install(model):
        raise RuntimeError("forced")
    assert model.scalar.requires_grad
    assert "forward" not in model.__dict__
    assert not controller._installed


def test_target_binding_uses_true_checkpoint_kekule_labels_and_deduplicates_regions():
    class Vocab:
        def __init__(self, tokens):
            self.tokens = tokens

        def indices_from_tokens(self, values):
            return [self.tokens.index(value) for value in values]

    model = SimpleNamespace(vocab=Vocab(["C", "N"]), vocab_charges=Vocab([0]), hparams={})
    cond = {
        "schema_version": "agenticgen.region-conditions.v1",
        "atomic_numbers": [6, 7],
        "formal_charges": [0, 0],
        "bond_types": [[0, 4], [4, 0]],
        "fixed_atom_input_slots": [0, 0, 1],
        "fixed_bond_pairs": [[0, 0], [0, 1], [1, 0]],
        "interaction_input_slots": [0],
        "interaction_coordinates_world": [[1, 2, 3]],
        "atom_maps": [3, 9],
    }
    model._agenticgen_condition_reference = {
        "atomics": torch.tensor([[1.0, 0.0], [0, 1]]),
        "charges": torch.ones((2, 1)),
        "bonds": torch.nn.functional.one_hot(torch.tensor([[0, 2], [2, 0]]), 5).float(),
    }
    result = targets_from_conditions(cond, model)
    assert result.bond_labels[0, 1] == 2
    assert result.atom_mask.sum() == 2
    assert torch.triu(result.bond_mask, 1).sum() == 1
    assert result.coordinate_mask.tolist() == [True, False]
    assert result.coordinates_world[0].tolist() == [1, 2, 3]
    model._agenticgen_condition_reference["atomics"] = torch.flip(
        model._agenticgen_condition_reference["atomics"], (0,)
    )
    with pytest.raises(ValueError, match="atom order"):
        targets_from_conditions(cond, model)


def test_invalid_condition_masks_and_nonfinite_reward_are_errors():
    tar = targets()
    no_anchors = dict(vars(tar))
    no_anchors["coordinate_mask"] = torch.zeros_like(tar.coordinate_mask)
    with pytest.raises(ValueError, match="Coordinate guidance"):
        RegionalReward(RegionTargets(**no_anchors), GuidanceConfig(coordinates=True))
    bad = dict(vars(tar))
    bad["bond_mask"] = tar.bond_mask.clone()
    bad["bond_mask"][1, 0] = False
    with pytest.raises(ValueError, match="symmetric"):
        RegionTargets(**bad)
    out = prediction()
    out["atomics"] = torch.full_like(out["atomics"], float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        RegionalReward(tar, GuidanceConfig())(out, out["coords"])
    with pytest.raises(ValueError, match="window"):
        GuidanceConfig(window=(0.2, 0.8))


def native_sparse_reference():
    """Native FLOWR encodes absent adjacency rows as zero vectors, not one-hot(0)."""

    class Vocab:
        def __init__(self, tokens):
            self.tokens = tokens

        def indices_from_tokens(self, values):
            return [self.tokens.index(value) for value in values]

    reference = {
        "atomics": torch.nn.functional.one_hot(torch.tensor([[0, 1, 0]]), 2).float(),
        "charges": torch.ones((1, 3, 1)),
        "bonds": torch.zeros((1, 3, 3, 5)),
        "mask": torch.ones((1, 3), dtype=torch.bool),
    }
    reference["bonds"][0, 0, 1, 1] = 1
    reference["bonds"][0, 1, 0, 1] = 1
    model = SimpleNamespace(
        vocab=Vocab(["C", "N"]),
        vocab_charges=Vocab([0]),
        hparams={"remove_aromaticity": True},
        _agenticgen_condition_reference=reference,
    )
    conditions = {
        "schema_version": "agenticgen.region-conditions.v1",
        "atomic_numbers": [6, 7, 6],
        "formal_charges": [0, 0, 0],
        "bond_types": [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        "fixed_atom_input_slots": [0, 1],
        "fixed_bond_pairs": [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]],
        "interaction_input_slots": [0],
        "interaction_coordinates_world": [[0.0, 0.0, 0.0]],
        "atom_maps": [3, 9, 42],
    }
    return model, conditions, reference


def test_native_zero_absent_bonds_bind_to_nobond_without_rewriting_reference_or_rng():
    model, conditions, reference = native_sparse_reference()
    original = {key: value.clone() for key, value in reference.items()}
    rng = torch.random.get_rng_state().clone()
    controller = build_controller(
        {"mode": "gradient", "guidance": {"family": "ce", "weight": 0}}, conditions, model
    )
    assert controller.targets.bond_labels.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert controller.targets.bond_mask[0, 2]
    assert not controller.targets.bond_mask.diagonal().any()
    with controller.install(model):
        pass
    for key, value in reference.items():
        assert torch.equal(value, original[key])
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize(
    "defect", ["atom_zero", "charge_zero", "bond_half", "bond_negative", "bond_lost"]
)
def test_demanded_native_categories_reject_invalid_values_instead_of_relaxing_all_validation(
    defect,
):
    model, conditions, reference = native_sparse_reference()
    if defect == "atom_zero":
        reference["atomics"][0, 0] = 0
    elif defect == "charge_zero":
        reference["charges"][0, 0] = 0
    elif defect == "bond_half":
        reference["bonds"][0, 0, 1] *= 0.5
    elif defect == "bond_negative":
        reference["bonds"][0, 0, 1, 0] = -1
        reference["bonds"][0, 0, 1, 1] = 2
    elif defect == "bond_lost":
        reference["bonds"][0, 0, 1] = 0
    with pytest.raises(ValueError, match="normalized|negative|connectivity"):
        targets_from_conditions(conditions, model)


def test_native_padding_and_diagonal_without_objective_are_not_categorical_targets():
    model, conditions, reference = native_sparse_reference()
    conditions["fixed_bond_pairs"] = [[0, 0], [0, 1], [1, 1]]
    reference["mask"][0, 2] = False
    reference["atomics"][0, 2] = 0
    reference["charges"][0, 2] = 0
    result = targets_from_conditions(conditions, model)
    assert result.atom_mask.tolist() == [True, True, False]
    assert result.bond_labels[0, 1] == 1
    conditions["fixed_bond_pairs"].append([0, 2])
    with pytest.raises(ValueError, match="absent from the native reference"):
        targets_from_conditions(conditions, model)
