"""Initialize edited poses without presenting RDKit coordinates as model output."""

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from agenticgen.contracts import GraphEditResult
from agenticgen.io import read_json
from agenticgen.pose_initialization import initialize_product_pose


def source_pose(path: Path, smiles: str) -> Path:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol = Chem.RemoveHs(mol)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return path


def edit(source, product):
    return GraphEditResult(status="accepted", source_smiles=source, product_mapped_smiles=product)


def coordinates(path):
    mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    return {
        a.GetAtomMapNum(): tuple(mol.GetConformer().GetAtomPosition(a.GetIdx()))
        for a in mol.GetAtoms()
    }


def test_deleted_atom_pose_uses_only_exact_map_coordinate_copy(tmp_path, monkeypatch):
    source = "[CH3:91][CH:7]([CH3:20])[OH:44]"
    product = "[CH3:91][CH2:7][OH:44]"
    sdf = source_pose(tmp_path / "source.sdf", source)

    def must_not_embed(*args, **kwargs):
        pytest.fail("Deletion-only initialization must not embed or move the source pose")

    monkeypatch.setattr(AllChem, "EmbedMolecule", must_not_embed)
    output = initialize_product_pose(source, edit(source, product), sdf, tmp_path / "product.sdf")
    expected = coordinates(sdf)
    assert coordinates(output) == {i: expected[i] for i in (91, 7, 44)}
    receipt = read_json(output.with_suffix(".initialization.json"))
    assert receipt["method"] == "exact_surviving_source_coordinate_copy"
    assert not receipt["is_flowr_output"] and not receipt["is_validated_bound_pose"]
    assert receipt["deleted_atom_maps"] == [20]
    assert receipt["max_surviving_source_displacement_angstrom"] == 0


def test_atom_substitution_keeps_original_position_and_records_change(tmp_path):
    source = "[CH3:91][CH2:7][OH:44]"
    product = "[CH3:91][NH:7][OH:44]"
    sdf = source_pose(tmp_path / "source.sdf", source)
    output = initialize_product_pose(source, edit(source, product), sdf, tmp_path / "product.sdf")
    assert coordinates(output) == coordinates(sdf)
    assert read_json(output.with_suffix(".initialization.json"))["substituted_atom_maps"] == [7]


def test_reconnected_graph_does_not_optimize_surviving_atom_positions(tmp_path):
    source = "[CH3:91][CH2:7][CH2:20][OH:44]"
    product = "[CH2:91]1[CH2:7][CH:20]1[OH:44]"
    sdf = source_pose(tmp_path / "source.sdf", source)
    output = initialize_product_pose(source, edit(source, product), sdf, tmp_path / "product.sdf")
    assert coordinates(output) == coordinates(sdf)


def test_added_atom_initialization_is_seeded_and_freezes_all_existing_heavy_atoms(tmp_path):
    source = "[CH3:91][CH2:7][OH:44]"
    product = "[CH3:91][CH:7]([OH:44])[F:501]"
    sdf = source_pose(tmp_path / "source.sdf", source)
    first = initialize_product_pose(
        source, edit(source, product), sdf, tmp_path / "first.sdf", seed=42
    )
    second = initialize_product_pose(
        source, edit(source, product), sdf, tmp_path / "second.sdf", seed=42
    )
    assert coordinates(first) == coordinates(second)
    assert {i: xyz for i, xyz in coordinates(first).items() if i != 501} == coordinates(sdf)
    receipt = read_json(first.with_suffix(".initialization.json"))
    assert receipt["added_atom_maps"] == [501]
    assert receipt["minimizer_status"] == 0
    assert receipt["method"].startswith("constrained_ETKDGv3")


def test_unmapped_source_pose_is_rejected_and_failure_receipt_is_kept(tmp_path):
    source = "[CH3:91][CH2:7][OH:44]"
    sdf = source_pose(tmp_path / "source.sdf", "CCO")
    output = tmp_path / "product.sdf"
    with pytest.raises(ValueError, match="unique positive"):
        initialize_product_pose(source, edit(source, source), sdf, output)
    assert not output.exists()
    assert read_json(output.with_suffix(".initialization.json"))["status"] == "failed"


def test_source_pose_with_wrong_graph_is_rejected(tmp_path):
    source = "[CH3:91][CH2:7][OH:44]"
    sdf = source_pose(tmp_path / "source.sdf", "[CH3:91][CH2:7][NH2:44]")
    with pytest.raises(ValueError, match="identity differs"):
        initialize_product_pose(source, edit(source, source), sdf, tmp_path / "product.sdf")


def test_invalid_synccraft_product_is_skipped_without_fake_pose(tmp_path):
    source = "[CH3:91][CH2:7][OH:44]"
    sdf = source_pose(tmp_path / "source.sdf", source)
    output = tmp_path / "product.sdf"
    rejected = GraphEditResult(status="rejected", source_smiles=source)
    with pytest.raises(ValueError, match="skipped, not repaired"):
        initialize_product_pose(source, rejected, sdf, output)
    assert not output.exists()
    assert read_json(output.with_suffix(".initialization.json"))["status"] == "failed"


def test_new_atom_embedding_failure_is_explicit(tmp_path, monkeypatch):
    source = "[CH3:91][CH2:7][OH:44]"
    product = "[CH3:91][CH:7]([OH:44])[F:501]"
    sdf = source_pose(tmp_path / "source.sdf", source)
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args, **kwargs: -1)
    output = tmp_path / "product.sdf"
    with pytest.raises(ValueError, match="could not initialize"):
        initialize_product_pose(source, edit(source, product), sdf, output)
    assert not output.exists()
    assert "ETKDG" in read_json(output.with_suffix(".initialization.json"))["reason"]
