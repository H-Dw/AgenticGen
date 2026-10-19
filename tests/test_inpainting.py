"""Mask, identity, native invocation and pose retention tests without a FLOWR model."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from agenticgen.contracts import GraphEditResult
from agenticgen.inpainting import (
    FLOWRInpaintingAdapter,
    collect_outputs,
    native_arguments,
    plan_masks,
    prepare_product_pose,
    preservation_report,
)
from agenticgen.inpainting_contracts import InpaintingConfig, InpaintingRequest
from agenticgen.io import read_json, sha256

SOURCE = "[CH3:91][CH2:7][OH:44]"
PRODUCT = "[CH3:91][CH:7]([OH:44])[F:501]"


def config(**kwargs):
    return InpaintingConfig(
        root="missing/source", python="missing/python", checkpoint="none", **kwargs
    )


def request(source=SOURCE, product=PRODUCT, **kwargs):
    return InpaintingRequest(
        case_id="test",
        proposal_id="proposal_0",
        source_mapped_smiles=source,
        graph_result=GraphEditResult(
            status="accepted", source_smiles=source, product_mapped_smiles=product
        ),
        **kwargs,
    )


def pose(smiles=PRODUCT):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(i * 1.2, i % 2 * 0.7, i % 3 * 0.4))
    mol.AddConformer(conf)
    return mol


def write_sdf(path, mol):
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return path


def test_mask_is_actual_graph_diff_and_not_map_id_minus_one():
    graph, fragment = plan_masks(request(interaction_atom_maps=(91,)), config())
    assert graph.fixed_input_slots == [0, 1, 2, 3]
    assert fragment.fixed_atom_maps == [7, 91, 501]
    assert fragment.optimized_atom_maps == [7, 501]
    assert fragment.cli_substructure_indices == [2]  # OH44 is the ONLY regenerable region.
    slots = {s.atom_map: s for s in fragment.atom_slots}
    assert slots[501].product_rdkit_index == 3
    assert slots[501].flowr_output_slot == 2  # Native fixed atoms move to the front.
    assert slots[44].flowr_output_slot is None  # Regenerated identity is not inherited.
    assert slots[91].source_rdkit_index == 0


def test_masks_follow_product_order_even_after_product_reordering():
    reordered = "[F:501][CH:7]([CH3:91])[OH:44]"
    _, fragment = plan_masks(request(product=reordered, interaction_atom_maps=(91,)), config())
    assert fragment.fixed_input_slots == [0, 1, 2]
    assert fragment.regenerate_input_slots == [3]
    assert {s.atom_map: s.flowr_output_slot for s in fragment.atom_slots}[91] == 2


def test_deletion_preserves_surviving_boundary_not_nonexistent_atoms():
    source = "[CH3:91][CH:7]([CH3:20])[OH:44]"
    _, fragment = plan_masks(request(source=source, product=SOURCE), config())
    assert fragment.deleted_atom_maps == [20]
    assert fragment.deletion_boundary_atom_maps == [7]
    assert fragment.fixed_atom_maps == [7]
    assert "cannot be fixed" in fragment.warnings[0]
    assert fragment.fixed_attachment_bonds[7] == ["SINGLE", "SINGLE"]


@pytest.mark.parametrize("interaction", [(7,), (999,)])
def test_strict_changed_or_unknown_interaction_skips_fragment_only(interaction):
    graph, fragment = plan_masks(
        request(interaction_atom_maps=interaction), config(interaction_policy="strict")
    )
    assert graph.status == "ready"
    assert fragment.status == "skipped"
    assert "conflicts" in fragment.reason or "Unknown source" in fragment.reason


def test_syncraft_policy_accepts_changed_interaction_in_edited_state():
    _, fragment = plan_masks(request(interaction_atom_maps=(7,)), config())
    assert fragment.status == "ready"
    assert fragment.interaction_changed_atom_maps == [7]
    assert fragment.interaction_surviving_atom_maps == [7]
    assert fragment.interaction_policy == "syncraft_compatible"
    assert any("not evidence" in warning for warning in fragment.warnings)


def test_syncraft_policy_records_deleted_interaction_instead_of_false_retention():
    source = "[CH3:91][CH:7]([CH3:20])[OH:44]"
    _, fragment = plan_masks(
        request(source=source, product=SOURCE, interaction_atom_maps=(20,)), config()
    )
    assert fragment.status == "ready"
    assert fragment.interaction_deleted_atom_maps == [20]
    assert fragment.interaction_surviving_atom_maps == []
    assert 20 not in fragment.fixed_atom_maps
    assert 7 in fragment.fixed_atom_maps


def test_no_fixed_or_no_variable_region_never_degrades_to_denovo():
    _, empty = plan_masks(request(product=SOURCE), config())
    assert empty.status == "skipped" and "No fixed" in empty.reason
    _, full = plan_masks(request(llm_optimized_atom_maps=(91, 7, 44, 501)), config())
    assert full.status == "skipped" and "No regenerable" in full.reason


def test_invalid_native_product_is_logged_without_loading_runtime(tmp_path):
    invalid = request().model_copy(
        update={
            "graph_result": GraphEditResult(
                status="rejected", source_smiles=SOURCE, reason="Invalid valence"
            )
        }
    )
    result = FLOWRInpaintingAdapter(config()).run(invalid, tmp_path, execute=True)
    assert result.status == "skipped"
    assert all(r.status == "skipped" for r in result.mode_results)
    assert "Invalid valence" in read_json(tmp_path / "result.json")["plans"][0]["reason"]
    assert not (tmp_path / "graph").exists()


def test_invalid_smiles_inside_nominally_accepted_product_is_skipped():
    plans = plan_masks(request(product="[CH5:7]"), config())
    assert all(p.status == "skipped" for p in plans)


def test_preview_requires_no_checkpoint_pose_pocket_or_gpu(tmp_path):
    result = FLOWRInpaintingAdapter(config()).run(request(), tmp_path)
    assert result.status == "planned"
    assert result.mode_results[0].status == "planned"
    assert read_json(tmp_path / "mask_plans.json")[1]["cli_substructure_indices"] == [0, 2]
    with pytest.raises(FileExistsError, match="new inpainting output"):
        FLOWRInpaintingAdapter(config()).run(request(), tmp_path)


def test_native_arguments_use_regeneration_indices_without_shell_or_graph_mix(tmp_path):
    req = request(pocket_pdb=str(tmp_path / "pocket with spaces.pdb"))
    graph, fragment = plan_masks(req, config())
    graph_args = native_arguments(config(), req, graph, tmp_path / "pose.sdf", tmp_path)
    frag_args = native_arguments(config(), req, fragment, tmp_path / "pose.sdf", tmp_path)
    assert "--graph_inpainting" in graph_args and "--substructure_inpainting" not in graph_args
    assert frag_args[-4:] == ["--substructure_inpainting", "--substructure", "0", "2"]
    assert "--sample_mol_sizes" not in frag_args
    assert "--canonicalize_conformer" not in frag_args
    assert "--filter_cond_substructure" not in frag_args
    assert "--final_inpaint" in graph_args and "--final_inpaint" in frag_args
    assert graph_args[graph_args.index("--graph_inpainting") + 1] == "harmonic"
    assert frag_args[1].endswith("pocket with spaces.pdb")


def test_prepare_mapped_pose_reorders_atoms_without_moving_coords(tmp_path):
    original = pose()
    reordered = Chem.RenumberAtoms(original, [3, 1, 0, 2])
    source = write_sdf(tmp_path / "reordered.sdf", reordered)
    prepared = prepare_product_pose(
        request(product_pose_sdf=str(source)), tmp_path / "prepared.sdf"
    )
    restored = Chem.SDMolSupplier(str(prepared), removeHs=True)[0]
    assert [a.GetAtomMapNum() for a in restored.GetAtoms()] == [91, 7, 44, 501]
    assert restored.GetConformer().GetAtomPosition(3).x == pytest.approx(3.6)


def test_symmetric_unmapped_pose_requires_explicit_identity(tmp_path):
    symmetrical = pose("[CH3:91][CH3:7]")
    for atom in symmetrical.GetAtoms():
        atom.SetAtomMapNum(0)
    sdf = write_sdf(tmp_path / "ambiguous.sdf", symmetrical)
    req = request(
        source="[CH3:91][CH3:7]",
        product="[CH3:91][CH3:7]",
        product_pose_sdf=str(sdf),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        prepare_product_pose(req, tmp_path / "prepared.sdf")


def test_pose_graph_mismatch_and_flat_conformer_are_not_repaired(tmp_path):
    sdf = write_sdf(tmp_path / "wrong.sdf", pose(SOURCE))
    with pytest.raises(ValueError, match="different heavy-atom count"):
        prepare_product_pose(request(product_pose_sdf=str(sdf)), tmp_path / "prepared.sdf")


def test_fragment_preservation_uses_fixed_first_native_slot_order():
    _, fragment = plan_masks(request(interaction_atom_maps=(91,)), config())
    reference = pose()
    generated = Chem.RenumberAtoms(reference, [0, 1, 3, 2])
    assert preservation_report(fragment, reference, generated, 0.05)["passed"]
    generated.GetConformer().SetAtomPosition(2, Point3D(8, 8, 8))
    report = preservation_report(fragment, reference, generated, 0.05)
    assert not report["passed"]
    assert any("map 501" in e for e in report["errors"])


def test_graph_can_move_coordinates_but_cannot_change_fixed_graph():
    graph = plan_masks(request(), config())[0]
    reference, generated = pose(), pose()
    generated.GetConformer().SetAtomPosition(0, Point3D(20, 20, 20))
    assert preservation_report(graph, reference, generated, 0.05)["passed"]
    generated.GetAtomWithIdx(3).SetAtomicNum(17)
    report = preservation_report(graph, reference, generated, 0.05)
    assert not report["passed"]
    assert any("identity differs" in e for e in report["errors"])


def test_raw_output_is_retained_and_new_fragment_atoms_get_new_identity(tmp_path):
    _, fragment = plan_masks(request(interaction_atom_maps=(91,)), config())
    reference = write_sdf(tmp_path / "reference.sdf", pose())
    raw = write_sdf(tmp_path / "native.sdf", Chem.RenumberAtoms(pose(), [0, 1, 3, 2]))
    raw_sha = sha256(raw)
    result = collect_outputs(fragment, reference, raw, tmp_path / "collected", config())
    assert result.status == "completed" and result.eligible_samples == 1
    assert sha256(raw) == raw_sha
    product = Chem.SDMolSupplier(result.poses[0].sdf_path, removeHs=True)[0]
    assert [a.GetAtomMapNum() for a in product.GetAtoms()][:3] == [91, 7, 501]
    assert product.GetAtomWithIdx(3).GetAtomMapNum() == 501 + 1000 + 3
    assert "V3000" in Path(result.poses[0].sdf_path).read_text(encoding="utf-8")
    assert len({a.GetAtomMapNum() for a in product.GetAtoms()}) == product.GetNumAtoms()
    assert result.poses[0].coordinate_frame == "pocket_input"


def test_moved_fixed_atom_is_reported_but_raw_pose_is_still_available(tmp_path):
    _, fragment = plan_masks(request(), config())
    generated = Chem.RenumberAtoms(pose(), [1, 3, 0, 2])
    generated.GetConformer().SetAtomPosition(0, Point3D(99, 99, 99))
    reference = write_sdf(tmp_path / "reference.sdf", pose())
    raw = write_sdf(tmp_path / "native.sdf", generated)
    result = collect_outputs(fragment, reference, raw, tmp_path / "collected", config())
    assert result.delivered_samples == 1 and result.eligible_samples == 0
    assert Path(result.poses[0].sdf_path).is_file()
    assert not read_json(Path(result.poses[0].preservation_report_path))["passed"]


def _native_ordered_pose(smiles, reference, plan):
    generated = pose(smiles)
    generated_maps = {a.GetAtomMapNum(): a.GetIdx() for a in generated.GetAtoms()}
    reference_maps = {a.GetAtomMapNum(): a.GetIdx() for a in reference.GetAtoms()}
    fixed = sorted(
        (slot for slot in plan.atom_slots if slot.fixed), key=lambda slot: slot.flowr_output_slot
    )
    for slot in fixed:
        generated.GetConformer().SetAtomPosition(
            generated_maps[slot.atom_map],
            reference.GetConformer().GetAtomPosition(reference_maps[slot.atom_map]),
        )
    order = [generated_maps[slot.atom_map] for slot in fixed]
    order.extend(i for i in range(generated.GetNumAtoms()) if i not in order)
    return Chem.RenumberAtoms(generated, order)


def test_interaction_only_external_connections_are_reported_without_hard_rejection():
    source = "[CH3:1][CH2:2][CH2:3][CH2:4][OH:5]"
    product = "[CH3:1][CH2:2][CH2:3][CH2:4][F:5]"
    _, fragment = plan_masks(
        request(source=source, product=product, interaction_atom_maps=(2, 3)), config()
    )
    reference = pose(product)
    generated = _native_ordered_pose("[CH3:2][CH:3]([CH3:1])[CH2:4][F:5]", reference, fragment)
    report = preservation_report(fragment, reference, generated, 0.05)
    assert report["passed"]
    assert {b["atom_map"] for b in report["free_boundary_changes"]} == {2, 3}
    assert {b["atom_map"] for b in report["checked_optimized_boundaries"]} == {5}
    assert all(b["matched"] for b in report["checked_optimized_boundaries"])
    assert report["warnings"] and report["max_fixed_coordinate_deviation_angstrom"] == 0


def test_actual_optimized_external_connections_remain_hard_constraints():
    _, fragment = plan_masks(request(interaction_atom_maps=(91,)), config())
    reference = pose()
    generated = _native_ordered_pose("[CH2:91]([OH:44])[CH2:7][F:501]", reference, fragment)
    report = preservation_report(fragment, reference, generated, 0.05)
    assert not report["passed"]
    assert any("Optimized boundary attachment differs at map 7" in e for e in report["errors"])
    assert {b["atom_map"] for b in report["free_boundary_changes"]} == {91}
    assert {b["atom_map"] for b in report["checked_optimized_boundaries"]} == {7, 501}


def test_deletion_boundary_remains_hard_even_when_also_an_interaction_atom():
    source = "[CH3:91][CH:7]([CH3:20])[OH:44]"
    _, fragment = plan_masks(
        request(source=source, product=SOURCE, interaction_atom_maps=(7,)), config()
    )
    reference = pose(SOURCE)
    generated = _native_ordered_pose("[CH3:7][CH2:91][OH:44]", reference, fragment)
    report = preservation_report(fragment, reference, generated, 0.05)
    # A deleted-boundary atom may not regain/change an optimized attachment.
    assert {b["atom_map"] for b in report["checked_optimized_boundaries"]} == {7}
    assert not report["passed"]
    assert any("Optimized boundary attachment differs at map 7" in e for e in report["errors"])


def test_prepared_pose_roundtrip_preserves_large_map_ids_and_mapped_graph(tmp_path):
    high_maps = "[CH3:1501][CH2:1502][OH:7777]"
    incoming = pose(high_maps)
    # The external pose is already V3000; the adapter must not downgrade it.
    source = tmp_path / "large_maps.sdf"
    with Chem.SDWriter(str(source)) as writer:
        writer.SetForceV3000(True)
        writer.write(Chem.RenumberAtoms(incoming, [2, 0, 1]))
    req = request(source=high_maps, product=high_maps, product_pose_sdf=str(source))
    prepared = prepare_product_pose(req, tmp_path / "prepared.sdf")
    loaded = Chem.SDMolSupplier(str(prepared), removeHs=True)[0]
    assert [a.GetAtomMapNum() for a in loaded.GetAtoms()] == [1501, 1502, 7777]
    assert Chem.MolToSmiles(loaded) == Chem.MolToSmiles(incoming)
    assert "V3000" in prepared.read_text(encoding="utf-8")


def test_multiple_regenerated_atoms_keep_distinct_large_namespace_ids(tmp_path):
    source = "[CH3:1][CH2:2][CH2:3][CH2:4][OH:5]"
    product = "[CH3:1][CH2:2][CH2:3][CH2:4][F:5]"
    _, fragment = plan_masks(
        request(source=source, product=product, interaction_atom_maps=(2, 3)), config()
    )
    reference = write_sdf(tmp_path / "reference.sdf", pose(product))
    generated = _native_ordered_pose(product, pose(product), fragment)
    raw = write_sdf(tmp_path / "native.sdf", generated)
    result = collect_outputs(fragment, reference, raw, tmp_path / "collected", config())
    loaded = Chem.SDMolSupplier(result.poses[0].sdf_path, removeHs=True)[0]
    assert [a.GetAtomMapNum() for a in loaded.GetAtoms()] == [2, 3, 5, 1008, 1009]
    assert len({a.GetAtomMapNum() for a in loaded.GetAtoms()}) == loaded.GetNumAtoms()
    assert Chem.MolToSmiles(loaded) == result.poses[0].mapped_smiles
