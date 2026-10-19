"""Regional endpoint guidance for the pinned native FLOWR sampler.

This is an approximate output-logit intervention, not an exact discrete Doob
transform. Current categorical argmax operations are never differentiated. A
separate, real-generator VJP maps the reward to current continuous coordinates.
PyTorch is an optional runtime dependency of this isolated FLOWR extension.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


def detached(value: Any) -> Any:
    """Stop earlier self-conditioning graphs, including nested tensor values."""
    if torch.is_tensor(value):
        value = value.detach()
        return value.clone() if torch.is_inference(value) else value
    if isinstance(value, dict):
        return {key: detached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detached(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detached(item) for item in value)
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Non-finite {name}")


@dataclass(frozen=True)
class GuidanceConfig:
    family: str = "ce"
    coordinates: bool = False
    weight: float = 1.0
    window: tuple[float, float] = (0.1, 0.9)
    coordinate_scale_angstrom: float = 0.5
    max_atom_step_angstrom: float = 0.025

    def __post_init__(self):
        if self.family not in {"ce", "brier"}:
            raise ValueError("Guidance family must be ce or brier")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Guidance weight must be finite and nonnegative")
        if tuple(self.window) != (0.1, 0.9):
            raise ValueError("The registered campaign fixes the window to [0.1,0.9)")
        if self.coordinate_scale_angstrom != 0.5 or self.max_atom_step_angstrom != 0.025:
            raise ValueError("The registered coordinate scale and step cap are frozen")

    def active(self, time: float) -> bool:
        # Native float32 addition can produce 0.099999986 or 0.9000001. The
        # campaign has an exact 100-step clock; rounding binds that clock.
        return self.weight > 0 and self.window[0] <= round(time, 6) < self.window[1]


@dataclass(frozen=True)
class RegionTargets:
    atom_labels: torch.Tensor
    charge_labels: torch.Tensor
    bond_labels: torch.Tensor
    atom_mask: torch.Tensor
    bond_mask: torch.Tensor
    coordinate_mask: torch.Tensor
    coordinates_world: torch.Tensor
    atom_maps: tuple[int, ...] = ()

    def __post_init__(self):
        n = self.atom_labels.numel()
        if self.atom_labels.shape != (n,) or self.charge_labels.shape != (n,):
            raise ValueError("Atom and charge targets must have one label per slot")
        if self.bond_labels.shape != (n, n) or self.bond_mask.shape != (n, n):
            raise ValueError("Bond targets and masks must be square slot matrices")
        if self.atom_mask.shape != (n,) or self.coordinate_mask.shape != (n,):
            raise ValueError("Region masks must have one entry per atom slot")
        if self.coordinates_world.shape != (n, 3):
            raise ValueError("Coordinate target must be an N by 3 world-Angstrom tensor")
        for value in (self.atom_labels, self.charge_labels, self.bond_labels):
            if value.dtype != torch.long:
                raise ValueError("Categorical target labels must be int64")
            if bool((value < 0).any()):
                raise ValueError("Negative categorical targets are not supported")
        for value in (self.atom_mask, self.bond_mask, self.coordinate_mask):
            if value.dtype != torch.bool:
                raise ValueError("Condition masks must be boolean")
        if not bool(self.atom_mask.any()):
            raise ValueError("Regional graph guidance needs at least one atom")
        if bool(self.bond_mask.diagonal().any()):
            raise ValueError("Self bonds must not be part of the graph target")
        if not torch.equal(self.bond_mask, self.bond_mask.T):
            raise ValueError("Undirected bond masks must be symmetric")
        if not torch.equal(self.bond_labels, self.bond_labels.T):
            raise ValueError("Undirected bond targets must be symmetric")
        if self.atom_maps and (len(self.atom_maps) != n or len(set(self.atom_maps)) != n):
            raise ValueError("Atom maps must bind distinct atoms to all slots")
        _finite(self.coordinates_world[self.coordinate_mask], "coordinate target")

    @property
    def size(self) -> int:
        return self.atom_labels.numel()

    def to(self, device: Any, dtype: torch.dtype) -> RegionTargets:
        return RegionTargets(
            self.atom_labels.to(device),
            self.charge_labels.to(device),
            self.bond_labels.to(device),
            self.atom_mask.to(device),
            self.bond_mask.to(device),
            self.coordinate_mask.to(device),
            self.coordinates_world.to(device=device, dtype=dtype),
            self.atom_maps,
        )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = int(mask.sum())
    if count == 0:
        raise ValueError("Cannot average an empty reward component")
    return value[:, mask].mean(dim=1)


def categorical_loss(logits: torch.Tensor, labels: torch.Tensor, family: str) -> torch.Tensor:
    """One scalar classification loss for each batch item and target slot."""
    _finite(logits, "endpoint logits")
    if int(labels.max()) >= logits.shape[-1]:
        raise ValueError("A categorical target is outside the checkpoint vocabulary")
    expanded = labels.unsqueeze(0).expand(logits.shape[:-1])
    if family == "ce":
        return -F.log_softmax(logits, dim=-1).gather(-1, expanded.unsqueeze(-1)).squeeze(-1)
    if family == "brier":
        one_hot = F.one_hot(expanded, num_classes=logits.shape[-1]).to(logits.dtype)
        return (F.softmax(logits, dim=-1) - one_hot).square().sum(-1)
    raise ValueError("Unknown categorical loss family")


class RegionalReward:
    """Balanced channel loss with unique atoms and unique undirected pairs.

    The four graph groups are atom, charge, present bond, and no-bond. Empty bond
    groups are omitted and reported, rather than inserted as fictitious zero loss.
    The optional coordinate loss is added to the balanced graph loss; it is not a
    guarantee of final interaction preservation. All are evaluated at the endpoint.
    """

    def __init__(self, targets: RegionTargets, config: GuidanceConfig):
        self.targets, self.config = targets, config
        if config.coordinates and not bool(targets.coordinate_mask.any()):
            raise ValueError("Coordinate guidance requested without surviving interaction anchors")

    def __call__(
        self, out: dict[str, torch.Tensor], coordinates_world: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tar = self.targets.to(out["atomics"].device, out["atomics"].dtype)
        b, n = out["atomics"].shape[:2]
        if n != tar.size or out["charges"].shape[:2] != (b, n):
            raise ValueError("Generated slots do not match regional graph targets")
        if out["bonds"].shape[:3] != (b, n, n) or coordinates_world.shape != (b, n, 3):
            raise ValueError("Endpoint bond/coordinate dimensions do not match targets")
        valid = out["mask"].bool()
        needed = tar.atom_mask | tar.coordinate_mask | tar.bond_mask.any(dim=-1)
        if not bool(valid[:, needed].all()):
            raise ValueError("A target atom slot is masked out of the native ligand")
        values = {
            "atom": _masked_mean(
                categorical_loss(out["atomics"], tar.atom_labels, self.config.family), tar.atom_mask
            ),
            "charge": _masked_mean(
                categorical_loss(out["charges"], tar.charge_labels, self.config.family),
                tar.atom_mask,
            ),
        }
        # Average mirrored logits first. The scalar loss counts each undirected
        # edge once, and its VJP gives identical updates to the two orientations.
        sym_logits = 0.5 * (out["bonds"] + out["bonds"].transpose(1, 2))
        bond_loss = categorical_loss(sym_logits, tar.bond_labels, self.config.family)
        unique_pairs = torch.triu(tar.bond_mask, diagonal=1)
        for name, select in (
            ("bond_present", unique_pairs & (tar.bond_labels > 0)),
            ("bond_absent", unique_pairs & (tar.bond_labels == 0)),
        ):
            if bool(select.any()):
                values[name] = _masked_mean(bond_loss, select)
        graph_loss = torch.stack(list(values.values()), dim=0).mean(dim=0)
        total_loss = graph_loss
        values["graph"] = graph_loss
        if self.config.coordinates:
            delta = (
                coordinates_world[:, tar.coordinate_mask]
                - tar.coordinates_world[tar.coordinate_mask]
            ) / self.config.coordinate_scale_angstrom
            # Radial pseudo-Huber: sqrt(1 + ||dx / scale||^2) - 1. This is smooth
            # at zero, rotationally covariant in the common receptor frame, and
            # does not introduce an artificial sqrt epsilon displacement.
            coordinate_loss = (torch.sqrt(1 + delta.square().sum(-1)) - 1).mean(dim=1)
            _finite(coordinate_loss, "coordinate reward")
            total_loss = total_loss + coordinate_loss
            values["coordinate"] = coordinate_loss
        _finite(total_loss, "regional reward")
        values["total"] = total_loss
        return -total_loss, values


def normalized_logit_update(
    out: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    targets: RegionTargets,
    weight: float,
    dt: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A common per-particle RMS on unique, directly supported logit entries."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Integration dt must be positive")
    device = out["atomics"].device
    atom = targets.atom_mask.to(device)
    pairs = torch.triu(targets.bond_mask.to(device), diagonal=1)
    support = {"atomics": atom, "charges": atom, "bonds": pairs}
    squared = out["atomics"].new_zeros(len(out["atomics"]))
    count = 0
    for key in ("atomics", "charges", "bonds"):
        gradient = gradients[key]
        _finite(gradient, f"{key} reward gradient")
        if gradient.shape != out[key].shape:
            raise ValueError("Logit gradient dimensions do not match native prediction")
        chosen = gradient[:, support[key], :]
        squared = squared + chosen.square().flatten(1).sum(dim=1)
        count += chosen[0].numel()
    rms = (squared / count).sqrt()
    denominator = rms.clamp_min(torch.finfo(rms.dtype).eps)
    probabilities, diagnostics = {}, {"logit_gradient_rms": rms}
    for key in ("atomics", "charges", "bonds"):
        gradient = gradients[key]
        divisor = denominator.reshape(-1, *([1] * (gradient.ndim - 1)))
        delta = weight * dt * gradient / divisor
        # Retain exact zero gradients, including a reward already at its optimum.
        delta = torch.where(gradient == 0, torch.zeros_like(delta), delta)
        logits = out[key].detach() + delta.detach()
        new = F.softmax(logits, dim=-1)
        log_native = F.log_softmax(out[key].detach(), dim=-1)
        log_new = F.log_softmax(logits, dim=-1)
        kl = (new * (log_new - log_native)).sum(-1)
        if bool(support[key].any()):
            diagnostics[f"{key}_kl"] = kl[:, support[key]].mean(dim=1)
        else:
            diagnostics[f"{key}_kl"] = rms * 0
        probabilities[key] = new
    return probabilities, diagnostics


def coordinate_displacement(
    gradient: torch.Tensor,
    native_delta: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
    coord_scale: float,
    cap_angstrom: float = 0.025,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale a real current-coordinate VJP by actual native step RMS and cap atoms."""
    if not math.isfinite(coord_scale) or coord_scale <= 0:
        raise ValueError("Native coordinate scaling must be positive")
    if gradient.shape != native_delta.shape or gradient.shape[:2] != mask.shape:
        raise ValueError("Current-coordinate VJP dimensions do not match native step")
    _finite(gradient, "current-coordinate reward gradient")
    _finite(native_delta, "native coordinate displacement")
    active = mask.bool().unsqueeze(-1)
    count = mask.sum(-1) * 3
    if not bool((count > 0).all()):
        raise ValueError("Empty native ligand")
    grad = gradient * active
    grad_rms = (grad.square().sum((1, 2)) / count).sqrt()
    native_rms = ((native_delta * active).square().sum((1, 2)) / count).sqrt()
    factor = weight * native_rms / grad_rms.clamp_min(torch.finfo(grad.dtype).eps)
    proposed = grad * factor[:, None, None]
    distances = proposed.norm(dim=-1) * coord_scale
    clipping = (cap_angstrom / distances.clamp_min(1e-30)).clamp(max=1)
    delta = proposed * clipping.unsqueeze(-1)
    lengths = delta.norm(dim=-1) * coord_scale
    return delta, {
        "coordinate_gradient_rms": grad_rms,
        "native_step_rms_angstrom": native_rms * coord_scale,
        "injected_rms_angstrom": (delta.square().sum((1, 2)) / count).sqrt() * coord_scale,
        "injected_max_atom_angstrom": lengths.amax(dim=1),
        "clipped_atom_fraction": ((clipping < 1) & mask.bool()).sum(-1) / mask.sum(-1),
        "minimum_clip_factor": clipping.masked_fill(~mask.bool(), 1).amin(dim=1),
    }


class FlowrGuidanceController:
    """Context-managed runtime extension; leaves the third-party checkout intact."""

    def __init__(self, targets: RegionTargets, config: GuidanceConfig):
        self.targets, self.config = targets, config
        self.reward = RegionalReward(targets, config)
        self.trace: list[dict[str, Any]] = []
        self._capture = None
        self._installed = False
        self.evidence: dict[str, Any] = {}

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "agenticgen.regional-reward-program.v1",
            "family": self.config.family,
            "coordinates": self.config.coordinates,
            "weight": self.config.weight,
            "window": list(self.config.window),
            "representation": "predicted_endpoint_logits_and_world_angstrom",
            "coordinate_scale_angstrom": self.config.coordinate_scale_angstrom,
            "max_atom_step_angstrom": self.config.max_atom_step_angstrom,
            "categorical_operator": "approximate_output_logit_gradient_then_native_CTMC",
            "coordinate_operator": "real_endpoint_to_current_VJP_then_native_RMS_scaling",
            "self_conditioning": "detached_native_previous_endpoint",
            "hard_constraints": False,
            "affinity_in_reward": False,
            "particle_resampling": False,
            "target_atom_maps": list(self.targets.atom_maps),
            "target_atom_count": int(self.targets.atom_mask.sum()),
            "target_undirected_pair_count": int(torch.triu(self.targets.bond_mask, 1).sum()),
            "coordinate_anchor_count": int(self.targets.coordinate_mask.sum()),
            "step_trace_count": len(self.trace),
            "evidence": self.evidence,
            "class_target_source": self.evidence.get(
                "target_provenance", "explicit_target_tensors"
            ),
            "aggregation": "mean_of_nonempty_atom_charge_present_bond_absent_bond_means",
            "coordinate_aggregation": "graph_loss_plus_mean_radial_pseudoHuber"
            if self.config.coordinates
            else None,
        }

    def reward_design_ir(self) -> dict[str, Any]:
        """Machine-readable experimental decision record, distinct from results."""
        return {
            "schema_version": "agenticgen.reward-design-ir.v1",
            "identity": self.evidence,
            "control_context": {
                "clock": "t=0 noise, t=1 endpoint; 100 linear steps",
                "coordinates": "current state, true endpoint VJP, receptor world frame in Angstrom",
                "categorical": "predicted endpoint logits; no derivative through current argmax",
                "native_update": "unchanged native CTMC after probability intervention",
            },
            "regions": {
                "graph": "union of surviving original interaction and actual edited/boundary slots; deduplicated",
                "coordinates": "surviving original Steer interaction coordinates only",
            },
            "objective_nodes": {
                "graph": {
                    "observable": self.config.family,
                    "target": "checkpoint-native encoding of edited product labels",
                    "units": "dimensionless",
                    "normalization": "per component count, then mean of nonempty components",
                    "gradient": "output logit gradient and real model current-coordinate VJP",
                    "validity": "fixed slot map, checkpoint-native aromatic encoding, common receptor frame",
                },
                "coordinate": {
                    "active": self.config.coordinates,
                    "observable": "sqrt(1 + ||endpoint_world - anchor_world||^2 / scale^2) - 1",
                    "scale_angstrom": self.config.coordinate_scale_angstrom,
                    "scale_origin": "user-approved exploratory constant, not calibrated physical tolerance",
                    "gradient": "real model endpoint-to-current coordinate VJP",
                },
            },
            "preservation_conditions": "soft surrogate only; independent final graph/contact tests required",
            "hard_constraints": [
                "valid target slots and finite derivatives",
                "0.025 Angstrom per-atom step cap",
            ],
            "terminal_utilities": [
                "pIC50",
                "Vina score_only",
                "SA",
                "strain",
                "chemical validity",
                "original contacts",
            ],
            "architecture_candidates": {
                "selected": "soft output-logit and coordinate hybrid, registered CE/Brier ablation",
                "hard_clamp": "separate inpainting comparison; not mixed with this soft experiment",
                "exact_discrete_rate_guidance": "not implemented; no exact conditional-rate claim",
                "affinity_reward": "excluded; independent terminal evaluator",
            },
            "sensitivity": {
                "ce_logit": "(one_hot(target)-p)/(nonempty_group_count*group_count)",
                "brier_logit": "2*J_softmax*(one_hot(target)-p)/(nonempty_group_count*group_count)",
                "coordinate": "-d/(anchor_count*scale^2*sqrt(1+||d||^2/scale^2)) before generator VJP",
                "optimum": "all supported p are target one-hot; if active, all anchor residuals are zero",
            },
            "execution_contract": self.provenance(),
            "feedback_contract": {
                "logs": "per-particle loss, raw gradients, native RMS, actual injection, clipping and KL",
                "failure": "nonfinite or disconnected derivatives fail task; no surrogate fallback",
                "reward_hacking": "correct marginal labels/anchors can coexist with invalid graph, wrong stereochemistry or lost contacts",
                "validation": "unit derivative checks passed separately; real-model and terminal results are external artifacts",
            },
        }

    @staticmethod
    def world_coordinates(model: Any, out: dict, pocket_batch: dict) -> torch.Tensor:
        coords = out["coords"] * float(model.coord_scale)
        if "complex" not in pocket_batch:
            raise ValueError("Native complex COM is required to bind original Steer coordinates")
        coms = [
            torch.as_tensor(system.com, device=coords.device) for system in pocket_batch["complex"]
        ]
        if len(coms) != len(coords):
            raise ValueError("Pocket COM batch does not match endpoint batch")
        return coords + torch.stack(coms).to(coords.dtype).reshape(-1, 1, 3)

    @contextmanager
    def install(self, model: Any):
        if self._installed:
            raise RuntimeError("A guidance controller cannot be installed twice")
        # This branch must do absolutely nothing: no forward wrapper, tensor
        # operation, parameter change, RNG use, or altered native output.
        if self.config.weight == 0:
            yield self
            return
        if bool(model.training):
            raise ValueError("Guidance requires the frozen model in eval mode")
        if bool(getattr(model, "graph_inpainting", False)) or bool(
            getattr(model, "inpainting_mode", False)
        ):
            raise ValueError("Independent soft guidance must not run with hard inpainting")
        original_forward, original_step = model.forward, model.integrator.step
        saved_forward = model.__dict__.get("forward")
        saved_step = model.integrator.__dict__.get("step")
        flags = [(param, param.requires_grad) for param in model.parameters()]
        for param, _ in flags:
            param.requires_grad_(False)
        controller = self

        def forward(_model, batch, pocket_batch, t, *args, **kwargs):
            time = float(t[0][0].detach().cpu())
            if not controller.config.active(time):
                controller._capture = None
                return original_forward(batch, pocket_batch, t, *args, **kwargs)
            if not bool(torch.allclose(t[0], t[0][0].expand_as(t[0]), atol=1e-6, rtol=0)):
                raise ValueError("Guidance requires a common sampling clock within a batch")
            with torch.inference_mode(False), torch.enable_grad():
                live_batch = detached(batch)
                x = batch["coords"].detach().clone().requires_grad_(True)
                live_batch["coords"] = x
                live_kwargs = detached(kwargs)
                out = original_forward(live_batch, detached(pocket_batch), t, *args, **live_kwargs)
                required = ("atomics", "charges", "bonds", "coords")
                if not all(out[key].requires_grad for key in required):
                    raise RuntimeError("Native endpoint output is not connected to autograd")
                world = controller.world_coordinates(model, out, pocket_batch)
                controller._capture = (x, out, world, time)
            return detached(out)

        def step(_integrator, curr, predicted, prior, times, step_size):
            time, dt = float(times[0][0].detach().cpu()), float(step_size)
            if not controller.config.active(time):
                controller._capture = None
                return original_step(curr, predicted, prior, times, step_size)
            if controller._capture is None:
                raise RuntimeError("An active guidance step has no live generator endpoint")
            x, out, world, captured_time = controller._capture
            if abs(captured_time - time) > 1e-5:
                raise RuntimeError("Guidance endpoint clock differs from native integrator clock")
            with torch.enable_grad():
                value, terms = controller.reward(out, world)
                variables = (out["atomics"], out["charges"], out["bonds"], x)
                grads = torch.autograd.grad(value.sum(), variables, allow_unused=False)
                categorical_grads = dict(
                    zip(("atomics", "charges", "bonds"), grads[:3], strict=True)
                )
                probabilities, cat_detail = normalized_logit_update(
                    out, categorical_grads, controller.targets, controller.config.weight, dt
                )
            guided = dict(predicted)
            guided.update(detached(probabilities))
            updated = original_step(curr, guided, prior, times, step_size)
            delta, coord_detail = coordinate_displacement(
                grads[-1].detach(),
                updated["coords"] - curr["coords"],
                curr["mask"],
                controller.config.weight,
                float(model.coord_scale),
                controller.config.max_atom_step_angstrom,
            )
            updated["coords"] = (updated["coords"] + delta) * curr["mask"].unsqueeze(-1)
            row = {"time": round(time, 6), "dt": dt}
            row.update(
                {f"loss_{key}": value.detach().cpu().tolist() for key, value in terms.items()}
            )
            row.update({key: value.detach().cpu().tolist() for key, value in cat_detail.items()})
            row.update({key: value.detach().cpu().tolist() for key, value in coord_detail.items()})
            controller.trace.append(row)
            controller._capture = None
            return detached(updated)

        model.forward = MethodType(forward, model)
        model.integrator.step = MethodType(step, model.integrator)
        self._installed = True
        try:
            yield self
        finally:
            if saved_forward is None:
                del model.__dict__["forward"]
            else:
                model.forward = saved_forward
            if saved_step is None:
                del model.integrator.__dict__["step"]
            else:
                model.integrator.step = saved_step
            for param, requires_grad in flags:
                param.requires_grad_(requires_grad)
            self._capture = None
            self._installed = False


def targets_from_conditions(conditions: dict[str, Any], model: Any) -> RegionTargets:
    """Bind the explicit, unreordered product slots to checkpoint vocabularies."""
    if conditions.get("schema_version") != "agenticgen.region-conditions.v1":
        raise ValueError("Unsupported regional condition schema")
    from rdkit import Chem

    atomic_numbers, charges = conditions["atomic_numbers"], conditions["formal_charges"]
    n = len(atomic_numbers)
    if n == 0 or len(charges) != n:
        raise ValueError("Condition categories are empty or misaligned")
    periodic = Chem.GetPeriodicTable()
    tokens = [periodic.GetElementSymbol(int(number)) for number in atomic_numbers]
    atom_labels = torch.tensor(model.vocab.indices_from_tokens(tokens), dtype=torch.long)
    charge_labels = torch.tensor(model.vocab_charges.indices_from_tokens(charges), dtype=torch.long)
    bonds = torch.tensor(conditions["bond_types"], dtype=torch.long)
    if bool((bonds > 4).any()):
        raise ValueError("Condition bond classes must be native 0/1/2/3/4 chemistry classes")
    slots = conditions.get("fixed_atom_input_slots")
    if slots is None:
        slots = sorted(
            set(conditions["interaction_input_slots"] + conditions["optimized_input_slots"])
        )
    # Duplicated evidence or overlap of I/E never increases objective weight.
    slots = sorted(set(slots))
    atom_mask = torch.zeros(n, dtype=torch.bool)
    if any(type(slot) is not int or not 0 <= slot < n for slot in slots):
        raise ValueError("Graph condition slot is outside the edited product")
    atom_mask[slots] = True
    pair_mask = torch.zeros((n, n), dtype=torch.bool)
    if "fixed_bond_pairs" not in conditions:
        raise ValueError("Explicit fixed_bond_pairs including no-bond targets are required")
    for pair in conditions["fixed_bond_pairs"]:
        i, j = pair[:2]
        if type(i) is not int or type(j) is not int or not 0 <= i < n or not 0 <= j < n:
            raise ValueError("Invalid graph condition bond pair")
        if i == j:
            continue  # Native hard masks can include diagonals; the loss cannot.
        pair_mask[i, j] = pair_mask[j, i] = True
    # The v2 checkpoint may remove aromaticity before encoding. Its actual
    # posterior uses Kekule single/double classes, whereas the RDKit condition
    # artifact retains class 4 for chemical identity checks. Never reward an
    # aromatic class which this checkpoint did not train on.
    reference = getattr(model, "_agenticgen_condition_reference", None)
    if reference is not None:
        encoded = {}
        valid = reference.get("mask", torch.ones(n, dtype=torch.bool))
        if valid.ndim == 2 and valid.shape[0] == 1:
            valid = valid[0]
        if valid.shape != (n,) or not bool(
            valid.bool().detach().cpu()[atom_mask | pair_mask.any(dim=-1)].all()
        ):
            raise ValueError("A regional condition target is absent from the native reference")
        for key, ndim in (("atomics", 2), ("charges", 2), ("bonds", 3)):
            tensor = reference[key]
            if tensor.ndim == ndim + 1 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != ndim or tensor.shape[:-1] != ((n, n) if key == "bonds" else (n,)):
                raise ValueError("Native encoded condition reference has incompatible slots")
            support = (pair_mask if key == "bonds" else atom_mask).to(tensor.device)
            selected = tensor[support]
            _finite(selected, "native encoded regional condition")
            if bool((selected < 0).any()):
                raise ValueError("Native condition categories cannot be negative")
            totals = selected.sum(-1)
            normalized = torch.isclose(totals, torch.ones_like(totals))
            if key == "bonds":
                # Native adjacency construction represents absent edges and
                # self/padding edges as all-zero vectors, not necessarily
                # one-hot(class 0). Native training obtains NOBOND with argmax=0.
                # Interpret that label without rewriting the native reference.
                normalized = normalized | (selected == 0).all(dim=-1)
            if not bool(normalized.all()):
                raise ValueError(f"Native {key} condition categories are not normalized")
            encoded[key] = tensor.argmax(-1).detach().cpu().long()
        if not torch.equal(
            atom_labels[atom_mask], encoded["atomics"][atom_mask]
        ) or not torch.equal(charge_labels[atom_mask], encoded["charges"][atom_mask]):
            raise ValueError(
                "Native condition preprocessing changed atom order, identities, or charges"
            )
        if not torch.equal((bonds > 0)[pair_mask], (encoded["bonds"] > 0)[pair_mask]):
            raise ValueError("Native condition preprocessing changed regional bond connectivity")
        # Unused labels, including diagonal and padded rows, are not objectives.
        # Keep a symmetric class-0 placeholder outside the explicit support.
        bonds = torch.where(pair_mask, encoded["bonds"], torch.zeros_like(bonds))
    elif bool(getattr(model, "hparams", {}).get("remove_aromaticity", False)):
        raise ValueError("Kekule checkpoint requires its actual native encoded condition reference")
    anchor_slots = conditions["interaction_input_slots"]
    anchor_values = conditions["interaction_coordinates_world"]
    if len(set(anchor_slots)) != len(anchor_slots) or len(anchor_values) != len(anchor_slots):
        raise ValueError("Interaction coordinates must match distinct original interaction slots")
    coordinates = torch.zeros((n, 3), dtype=torch.float32)
    coordinate_mask = torch.zeros(n, dtype=torch.bool)
    for slot, xyz in zip(anchor_slots, anchor_values, strict=True):
        if type(slot) is not int or not 0 <= slot < n or xyz is None or len(xyz) != 3:
            raise ValueError("Original Steer interaction coordinate is missing or invalid")
        coordinate_mask[slot] = True
        coordinates[slot] = torch.tensor(xyz)
    return RegionTargets(
        atom_labels,
        charge_labels,
        bonds,
        atom_mask,
        pair_mask,
        coordinate_mask,
        coordinates,
        tuple(conditions["atom_maps"]),
    )


def build_controller(task: dict[str, Any], conditions: dict[str, Any], model: Any):
    """Worker entry point; constructing the zero-weight controller never uses RNG."""
    spec = task.get("guidance", {})
    config = GuidanceConfig(
        family=spec.get("family", "ce"),
        coordinates=spec.get("coordinates", False),
        weight=float(spec.get("weight", 0.0 if task["mode"] == "unguided" else 1.0)),
        window=tuple(spec.get("window", (0.1, 0.9))),
    )
    controller = FlowrGuidanceController(targets_from_conditions(conditions, model), config)
    controller.evidence = {
        "conditions_sha256": hashlib.sha256(
            json.dumps(conditions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "source_pose_sdf": task.get("source_pose_sdf"),
        "product_pose_sdf": task.get("product_pose_sdf"),
        "checkpoint": task.get("checkpoint"),
        "case_id": task.get("case_id"),
        "target_provenance": "native_encoded_product_posterior"
        if getattr(model, "_agenticgen_condition_reference", None) is not None
        else "vocabulary_encoding_of_chemical_conditions",
    }
    return controller


def probe_model_reward(
    controller: FlowrGuidanceController,
    model: Any,
    batch: dict[str, Any],
    pocket_batch: dict[str, Any],
    times: list[torch.Tensor],
    cond_batch: dict[str, Any] | None = None,
    *,
    epsilon: float = 0.001,
    relative_tolerance: float = 0.1,
    **forward_kwargs,
) -> dict[str, Any]:
    """Check a real model's local VJP and all three endpoint-logit derivatives.

    The caller supplies an actual noisy ligand state, the corresponding native
    pocket, and native detached self-conditioning. Random states are restored
    around every forward and after the probe. Finite-difference failure is an
    explicit preflight failure, never a reason to silently substitute a gradient.
    This does not test chemical validity or downstream reward usefulness.
    """
    if model.training or controller._installed:
        raise ValueError("The preflight requires an eval model without an installed controller")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("Finite-difference epsilon must be finite and positive")
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore_rng():
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    def predict(x):
        restore_rng()
        value = detached(batch)
        value["coords"] = x
        return model(
            value,
            detached(pocket_batch),
            times,
            cond_batch=detached(cond_batch),
            training=False,
            **detached(forward_kwargs),
        )

    flags = [(param, param.requires_grad) for param in model.parameters()]
    for param, _ in flags:
        param.requires_grad_(False)
    try:
        with torch.inference_mode(False), torch.enable_grad():
            x = batch["coords"].detach().clone().requires_grad_(True)
            out = predict(x)
            world = controller.world_coordinates(model, out, pocket_batch)
            reward, terms = controller.reward(out, world)
            variables = (out["atomics"], out["charges"], out["bonds"], x)
            grads = torch.autograd.grad(reward.sum(), variables, allow_unused=False)
        records = {}

        def compare(name, analytic, plus, minus):
            numerical = (float(plus) - float(minus)) / (2 * epsilon)
            analytic = float(analytic)
            error = abs(numerical - analytic)
            relative = error / max(abs(numerical), abs(analytic), 1e-8)
            passed = math.isfinite(relative) and (relative <= relative_tolerance or error <= 1e-5)
            records[name] = {
                "analytic": analytic,
                "finite_difference": numerical,
                "absolute_error": error,
                "relative_error": relative,
                "passed": passed,
            }

        with torch.no_grad():
            # The gradient direction avoids accidental orthogonality in a tiny
            # pilot and does not consume randomness.
            grad = grads[-1]
            norm = grad.norm()
            if not bool(torch.isfinite(norm)) or float(norm) == 0:
                raise RuntimeError("Real-generator current-coordinate VJP is zero or non-finite")
            direction = grad / norm
            plus_out = predict(x.detach() + epsilon * direction)
            minus_out = predict(x.detach() - epsilon * direction)
            plus, _ = controller.reward(
                plus_out, controller.world_coordinates(model, plus_out, pocket_batch)
            )
            minus, _ = controller.reward(
                minus_out, controller.world_coordinates(model, minus_out, pocket_batch)
            )
            compare("current_coordinate_VJP", norm, plus.sum(), minus.sum())
            # Probe-only copies avoid cancellation of a weak head derivative
            # inside a much larger float32 total reward. Both the perturbation
            # and complete scalar loss use float64 on CPU; native model calls,
            # autograd derivatives, production loss, epsilon and tolerance stay
            # unchanged. Merely summing already-rounded float32 losses in double
            # would not recover the missing difference.
            logit_reference = {
                key: value.detach().to(device="cpu", dtype=torch.float64)
                if key in {"atomics", "charges", "bonds", "coords"}
                else value.detach().cpu()
                for key, value in out.items()
                if key in {"atomics", "charges", "bonds", "coords", "mask"}
            }
            world_double = world.detach().to(device="cpu", dtype=torch.float64)
            for key, grad in zip(("atomics", "charges", "bonds"), grads[:3], strict=True):
                direction_gradient = grad.detach().to(device="cpu", dtype=torch.float64)
                norm = direction_gradient.norm()
                if float(norm) == 0:
                    records[key] = {"passed": False, "reason": "zero_logit_gradient"}
                    continue
                direction = direction_gradient / norm
                plus_out, minus_out = dict(logit_reference), dict(logit_reference)
                plus_out[key] = plus_out[key] + epsilon * direction
                minus_out[key] = minus_out[key] - epsilon * direction
                plus, _ = controller.reward(plus_out, world_double)
                minus, _ = controller.reward(minus_out, world_double)
                compare(key, norm, plus.sum(), minus.sum())
        report = {
            "schema_version": "agenticgen.guidance-model-preflight.v1",
            "epsilon": epsilon,
            "relative_tolerance": relative_tolerance,
            "time": float(times[0][0]),
            "batch_size": len(batch["coords"]),
            "reward": reward.detach().cpu().tolist(),
            "loss_terms": {key: value.detach().cpu().tolist() for key, value in terms.items()},
            "derivatives": records,
            "passed": all(row["passed"] for row in records.values()),
            "validation_scope": "local_real_generator_VJP_and_output_logits_only",
            "finite_difference_arithmetic": {
                "current_coordinate_VJP": "unchanged_native_model_and_loss_dtype",
                "output_logits": "float64_CPU_loss_and_perturbation_on_detached_copies",
                "analytic_gradients": "unchanged_native_autograd_dtype",
                "sampling_changed": False,
            },
        }
        if not report["passed"]:
            raise RuntimeError(f"Guidance real-model finite-difference preflight failed: {report}")
        return report
    finally:
        restore_rng()
        for param, requires_grad in flags:
            param.requires_grad_(requires_grad)
