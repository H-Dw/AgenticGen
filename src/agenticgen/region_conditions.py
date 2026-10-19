"""Independent graph, boundary and coordinate conditions for controlled campaigns.

No embedding is performed. A readable edited graph is enough to construct the
graph-only arms. Pose targets, when used, come only from the mapped source pose.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rdkit import Chem

from agenticgen.chemistry import molecule
from agenticgen.contracts import GraphEditResult
from agenticgen.inpainting import _mapped_atoms, _pose_graph_matches, plan_masks
from agenticgen.inpainting_contracts import InpaintingConfig, InpaintingRequest
from agenticgen.io import sha256

CAMPAIGN_MODES = ("graph", "fragment_graph", "fragment_graph_pose", "gradient", "unguided")


def build_region_conditions(
    source_mapped_smiles: str,
    graph_result: GraphEditResult | dict,
    interaction_atom_maps: Iterable[int],
    source_pose_sdf: Path | None,
    product_pose_sdf: Path | None = None,
) -> dict[str, Any]:
    source_pose_sdf = Path(source_pose_sdf) if source_pose_sdf is not None else None
    product_pose_sdf = Path(product_pose_sdf) if product_pose_sdf is not None else None
    graph = (
        graph_result
        if isinstance(graph_result, GraphEditResult)
        else GraphEditResult.model_validate(graph_result)
    )
    request = InpaintingRequest(
        case_id="campaign",
        proposal_id="conditions",
        source_mapped_smiles=source_mapped_smiles,
        graph_result=graph,
        interaction_atom_maps=tuple(interaction_atom_maps),
    )
    graph_plan, local_plan = plan_masks(
        request, InpaintingConfig(root="", python="", checkpoint="", modes=("graph", "fragment"))
    )
    base = {
        "schema_version": "agenticgen.region-conditions.v1",
        "status": "ready",
        "source_mapped_smiles": source_mapped_smiles,
        "product_mapped_smiles": graph.product_mapped_smiles,
        "legacy_mask_plan": local_plan.model_dump(mode="json"),
        "warnings": list(local_plan.warnings),
        "source_pose_sdf": str(source_pose_sdf.resolve()) if source_pose_sdf else None,
        "product_pose_sdf": str(product_pose_sdf.resolve()) if product_pose_sdf else None,
        "product_coordinates_are_condition_targets": False,
        "source_pose_sha256": None,
        "interaction_coordinates_world": None,
        "pose_condition_error": None,
        "categorical_representation": "RDKit chemical classes; runtime uses native checkpoint encoding",
    }
    if graph_plan.status == "skipped":
        return {**base, "status": "skipped", "reason": graph_plan.reason}
    product = molecule(graph.product_mapped_smiles)
    atoms = _mapped_atoms(product)
    atom_maps = [a.GetAtomMapNum() for a in product.GetAtoms()]
    n = len(atom_maps)
    interaction = [atoms[i].GetIdx() for i in local_plan.interaction_surviving_atom_maps]
    interaction.sort()
    optimized = sorted(atoms[i].GetIdx() for i in local_plan.optimized_atom_maps)
    q = sorted(set(interaction) | set(optimized))
    q_set, e_set = set(q), set(optimized)
    pairs = [
        [i, j]
        for i in range(n)
        for j in range(i, n)
        if (i in q_set and j in q_set) or i in e_set or j in e_set
    ]
    bonds = [[0 for _ in range(n)] for _ in range(n)]
    classes = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for bond in product.GetBonds():
        kind = classes.get(bond.GetBondType())
        if kind is None:
            return {**base, "status": "skipped", "reason": "Unsupported molecular bond class"}
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[i][j] = bonds[j][i] = kind
    categorical = {
        "atomic_numbers": [a.GetAtomicNum() for a in product.GetAtoms()],
        "formal_charges": [a.GetFormalCharge() for a in product.GetAtoms()],
        "bond_orders": bonds,
    }
    base.update(
        atom_maps=atom_maps,
        atomic_numbers=categorical["atomic_numbers"],
        formal_charges=categorical["formal_charges"],
        bond_types=bonds,
        categorical_targets=categorical,
        interaction_input_slots=interaction,
        optimized_input_slots=optimized,
        fixed_atom_input_slots=q,
        fixed_bond_pairs=pairs,
        fixed_bond_values=[bonds[i][j] for i, j in pairs],
        deleted_atom_maps=local_plan.deleted_atom_maps,
        deletion_boundary_atom_maps=local_plan.deletion_boundary_atom_maps,
        interaction_deleted_atom_maps=local_plan.interaction_deleted_atom_maps,
        interaction_changed_atom_maps=local_plan.interaction_changed_atom_maps,
        atom_slots=[
            {
                **slot.model_dump(mode="json"),
                "flowr_output_slot": slot.input_slot,
                "fixed": slot.input_slot in q_set,
            }
            for slot in local_plan.atom_slots
        ],
        slot_policy="No reordering; free output slots do not inherit source atom identities",
        graph_boundary_policy="Q induced pairs plus E incident rows, including absent bonds",
    )
    if "Unknown source interaction" in (local_plan.reason or ""):
        base["local_condition_error"] = local_plan.reason
    if source_pose_sdf:
        try:
            source_pose_sdf = source_pose_sdf.resolve()
            supplier = Chem.SDMolSupplier(str(source_pose_sdf), removeHs=True)
            if len(supplier) != 1 or supplier[0] is None:
                raise ValueError("Source pose must contain exactly one readable molecule")
            pose = supplier[0]
            source = molecule(source_mapped_smiles)
            source_atoms, pose_atoms = _mapped_atoms(source), _mapped_atoms(pose)
            if set(source_atoms) != set(pose_atoms) or not _pose_graph_matches(source, pose):
                raise ValueError("Source pose atom-map identities or graph do not match the source")
            # Check correspondence directly, not just a symmetry-related graph match.
            for i, a in source_atoms.items():
                if (
                    a.GetAtomicNum() != pose_atoms[i].GetAtomicNum()
                    or a.GetFormalCharge() != pose_atoms[i].GetFormalCharge()
                ):
                    raise ValueError("Mapped source pose atom identity differs")
                for j in source_atoms:
                    x = source.GetBondBetweenAtoms(a.GetIdx(), source_atoms[j].GetIdx())
                    y = pose.GetBondBetweenAtoms(pose_atoms[i].GetIdx(), pose_atoms[j].GetIdx())
                    if (str(x.GetBondType()) if x else None) != (
                        str(y.GetBondType()) if y else None
                    ):
                        raise ValueError("Mapped source pose connectivity differs")
            if not pose.GetNumConformers() or not pose.GetConformer().Is3D():
                raise ValueError("Source pose has no 3D conformer")
            coords = [
                list(pose.GetConformer().GetAtomPosition(pose_atoms[atom_maps[i]].GetIdx()))
                for i in interaction
            ]
            if not all(math.isfinite(v) for xyz in coords for v in xyz):
                raise ValueError("Interaction coordinates are non-finite")
            base.update(
                interaction_coordinates_world=coords, source_pose_sha256=sha256(source_pose_sdf)
            )
        except (ValueError, OSError, RuntimeError) as error:
            base["pose_condition_error"] = str(error)
            base["warnings"].append(
                "Pose conditions unavailable; pure graph conditions remain usable"
            )
    elif interaction:
        base["pose_condition_error"] = "No source pose was supplied"
    else:
        base["interaction_coordinates_world"] = []
    return base


def mode_channel_masks(conditions: dict, mode: str) -> dict:
    """Select independent channels. An empty/zero-degree arm is never relabeled de novo."""
    if mode not in CAMPAIGN_MODES:
        raise ValueError("Unknown campaign mode")
    if conditions.get("status") != "ready":
        return {
            "status": "not_applicable",
            "reason": conditions.get("reason", "Invalid product"),
            "mode": mode,
        }
    n = len(conditions["atom_maps"])
    if mode == "graph":
        atoms = list(range(n))
        pairs = [[i, j] for i in range(n) for j in range(i, n)]
        coordinates = []
    elif mode in {"gradient", "unguided"}:
        atoms, pairs, coordinates = [], [], []
    else:
        if conditions.get("local_condition_error"):
            return {
                "status": "not_applicable",
                "reason": conditions["local_condition_error"],
                "mode": mode,
            }
        atoms = conditions["fixed_atom_input_slots"]
        pairs = conditions["fixed_bond_pairs"]
        coordinates = conditions["interaction_input_slots"] if mode == "fragment_graph_pose" else []
        if mode == "fragment_graph_pose" and not coordinates:
            return {
                "status": "not_applicable",
                "reason": "No surviving original interaction coordinates; pose arm cannot degrade to graph-only",
                "mode": mode,
            }
        if not atoms and not pairs:
            return {
                "status": "not_applicable",
                "reason": "Empty local graph condition; no silent de novo fallback",
                "mode": mode,
            }
        if coordinates and conditions.get("interaction_coordinates_world") is None:
            return {
                "status": "not_applicable",
                "reason": conditions.get("pose_condition_error")
                or "Interaction pose targets unavailable",
                "mode": mode,
            }
    free_atoms = n - len(atoms)
    free_pairs = n * (n + 1) // 2 - len(pairs)
    free_coords = 3 * (n - len(coordinates))
    status = "ready" if free_atoms + free_pairs + free_coords else "not_applicable"
    return {
        "status": status,
        "mode": mode,
        "reason": "No generation degrees of freedom" if status != "ready" else None,
        "fixed_atom_input_slots": atoms,
        "fixed_bond_pairs": pairs,
        "fixed_coordinate_input_slots": coordinates,
        "free_atom_slots": free_atoms,
        "free_bond_pairs": free_pairs,
        "free_coordinate_degrees": free_coords,
        "continuous_clock_fixed_slots": coordinates,
        "discrete_clock_fixed_slots": atoms,
        "coordinate_target_origin": "original source pose only",
        "edge_clock_policy": "checkpoint has node clocks; fixed edge values are enforced independently",
    }


class RegionConditioner:
    """Channel enforcement for native model calls, without a reference-coordinate leak.

    Tensor construction uses the supplied native tensors' factory methods, allowing
    lightweight tensor doubles in unit tests without importing the FLOWR environment.
    """

    def __init__(self, channels: dict, reference: dict):
        self.channels, self.reference = channels, reference
        batch, n = reference["mask"].shape
        atoms = set(channels["fixed_atom_input_slots"])
        coords = set(channels["fixed_coordinate_input_slots"])
        self.atom_mask = (
            reference["mask"].new_tensor([[i in atoms for i in range(n)]] * batch).bool()
        )
        self.coordinate_mask = (
            reference["mask"].new_tensor([[i in coords for i in range(n)]] * batch).bool()
        )
        pairs = {tuple(pair) for pair in channels["fixed_bond_pairs"]}
        self.bond_mask = (
            reference["mask"]
            .new_tensor(
                [[[(min(i, j), max(i, j)) in pairs for j in range(n)] for i in range(n)]] * batch
            )
            .bool()
        )

    def apply(self, prediction: dict, *, feature_keys=None, overwrite_with_zeros=False) -> dict:
        result = dict(prediction)
        keys = (
            list(feature_keys)
            if feature_keys is not None
            else ["atomics", "charges", "bonds", "hybridization"]
        )
        # Position conditions are an independent channel even when native graph
        # feature_keys does not mention coords (notably final_inpaint).
        keys = [key for key in keys if key != "coords"]
        # Native SC feature_keys omit charge/additional atom features. When those
        # tensors exist, they still belong to the fixed local graph condition.
        # This also covers residual/self-conditioned charge checkpoints.
        keys = list(dict.fromkeys([*keys, "charges", "hybridization", "coords"]))
        for key in keys:
            if key not in result or result[key] is None or key not in self.reference:
                continue
            mask = (
                self.coordinate_mask
                if key == "coords"
                else self.bond_mask
                if key == "bonds"
                else self.atom_mask
            )
            value = result[key].clone()
            target = self.reference[key]
            value[mask] = 0 if overwrite_with_zeros and key != "coords" else target[mask]
            result[key] = value
        result["fragment_mask"] = self.atom_mask
        result["fragment_mode"] = [self.channels["mode"]] * self.atom_mask.shape[0]
        return result

    def times(self, times: list, lig_mask) -> list:
        n = lig_mask.shape[1]
        continuous = times[0].reshape(-1, 1, 1).expand(-1, n, -1).clone()
        discrete = times[1].reshape(-1, 1, 1).expand(-1, n, -1).clone()
        continuous[self.coordinate_mask] = 1.0
        discrete[self.atom_mask] = 1.0
        return [continuous, discrete]
