"""Independent graph/pose condition planning without importing FLOWR or PyTorch."""

from pathlib import Path

from rdkit import Chem

from agenticgen.contracts import GraphEditResult
from agenticgen.region_conditions import build_region_conditions, mode_channel_masks

SOURCE = "[CH3:91][CH2:7][OH:44]"
PRODUCT = "[CH3:91][CH:7]([OH:44])[F:501]"


def conditions(source=SOURCE, product=PRODUCT, interactions=(91,), **kwargs):
    return build_region_conditions(
        source,
        GraphEditResult(status="accepted", source_smiles=source, product_mapped_smiles=product),
        interactions,
        kwargs.get("source_pose_sdf"),
        kwargs.get("product_pose_sdf"),
    )


def write_source_pose(path: Path, source=SOURCE):
    mol = Chem.MolFromSmiles(source)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, (10.0 + i, 20.0 - i, -5.0 + i))
    mol.AddConformer(conformer)
    with Chem.SDWriter(str(path)) as writer:
        writer.SetForceV3000(True)
        writer.write(mol)
    return path


def test_graph_arms_do_not_require_product_pose_or_embedding(tmp_path):
    result = conditions(product_pose_sdf=tmp_path / "does-not-exist.sdf")
    assert result["status"] == "ready"
    assert mode_channel_masks(result, "graph")["status"] == "ready"
    assert mode_channel_masks(result, "fragment_graph")["status"] == "ready"
    pose = mode_channel_masks(result, "fragment_graph_pose")
    assert pose["status"] == "not_applicable"
    assert "No source pose" in pose["reason"]


def test_i_e_union_preserves_product_order_without_fixed_first_reordering(tmp_path):
    result = conditions(source_pose_sdf=write_source_pose(tmp_path / "source.sdf"))
    assert result["interaction_input_slots"] == [0]
    assert result["optimized_input_slots"] == [1, 3]
    assert result["fixed_atom_input_slots"] == [0, 1, 3]
    assert result["atom_maps"] == [91, 7, 44, 501]
    assert [s["flowr_output_slot"] for s in result["atom_slots"]] == [0, 1, 2, 3]
    assert result["interaction_coordinates_world"] == [[10.0, 20.0, -5.0]]
    assert not result["product_coordinates_are_condition_targets"]


def test_local_pairs_fix_optimized_rows_and_q_internal_nonbonds():
    result = conditions()
    pairs = {tuple(pair) for pair in result["fixed_bond_pairs"]}
    assert (0, 3) in pairs  # Q internal nonbond: keep absent.
    assert (1, 2) in pairs  # E -> variable boundary edge: keep edit.
    assert (2, 3) in pairs  # E -> variable boundary nonbond: prevent restoration.
    assert (2, 2) not in pairs  # Free variable self pair is outside the local condition.
    assert result["categorical_targets"]["bond_orders"][0][3] == 0


def test_three_arms_have_different_coordinate_channels_only(tmp_path):
    result = conditions(source_pose_sdf=write_source_pose(tmp_path / "source.sdf"))
    graph = mode_channel_masks(result, "graph")
    local = mode_channel_masks(result, "fragment_graph")
    pose = mode_channel_masks(result, "fragment_graph_pose")
    assert graph["fixed_atom_input_slots"] == [0, 1, 2, 3]
    assert graph["fixed_coordinate_input_slots"] == []
    assert local["fixed_atom_input_slots"] == pose["fixed_atom_input_slots"] == [0, 1, 3]
    assert local["fixed_bond_pairs"] == pose["fixed_bond_pairs"]
    assert local["fixed_coordinate_input_slots"] == []
    assert pose["fixed_coordinate_input_slots"] == [0]
    assert pose["free_coordinate_degrees"] == 9  # E remains geometrically free.


def test_gradient_and_unguided_never_acquire_hard_conditions():
    for mode in ("gradient", "unguided"):
        mask = mode_channel_masks(conditions(), mode)
        assert mask["status"] == "ready"
        assert mask["fixed_atom_input_slots"] == []
        assert mask["fixed_bond_pairs"] == []
        assert mask["fixed_coordinate_input_slots"] == []


def test_deleted_interaction_atoms_are_released_not_restored(tmp_path):
    source = "[CH3:91][CH:7]([CH3:20])[OH:44]"
    result = conditions(
        source=source,
        product=SOURCE,
        interactions=(20,),
        source_pose_sdf=write_source_pose(tmp_path / "source.sdf", source),
    )
    assert result["interaction_deleted_atom_maps"] == [20]
    assert result["interaction_input_slots"] == []
    assert result["optimized_input_slots"] == [1]
    assert 20 not in result["atom_maps"]
    assert result["legacy_mask_plan"]["deletion_boundary_atom_maps"] == [7]
    assert mode_channel_masks(result, "fragment_graph_pose")["status"] == "not_applicable"
    assert "cannot degrade" in mode_channel_masks(result, "fragment_graph_pose")["reason"]
    assert mode_channel_masks(result, "fragment_graph")["status"] == "ready"


def test_all_graph_fixed_still_has_coordinate_freedom_and_zero_dof_is_not_applicable(tmp_path):
    result = conditions(
        product=SOURCE,
        interactions=(91, 7, 44),
        source_pose_sdf=write_source_pose(tmp_path / "source.sdf"),
    )
    assert mode_channel_masks(result, "fragment_graph")["status"] == "ready"
    zero = mode_channel_masks(result, "fragment_graph_pose")
    assert zero["status"] == "not_applicable" and "No generation degrees" in zero["reason"]


def test_unparseable_graph_skips_all_arms():
    result = build_region_conditions(
        SOURCE,
        GraphEditResult(status="rejected", source_smiles=SOURCE, reason="bad graph"),
        (),
        None,
    )
    assert result["status"] == "skipped"
    assert all(
        mode_channel_masks(result, mode)["status"] == "not_applicable"
        for mode in ("graph", "fragment_graph", "gradient", "unguided")
    )
