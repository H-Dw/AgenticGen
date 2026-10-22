from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from agenticgen.workers.plip_alignment import pose_graph_smiles
from agenticgen.workers.syncraft_geometry import load_input_pose, prepare_and_score_pose


def test_score_only_keeps_supplied_heavy_atoms_and_parses_vina12(tmp_path, monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]"))
    AllChem.EmbedMolecule(mol, randomSeed=9)
    pose = tmp_path / "pose.sdf"
    with Chem.SDWriter(str(pose)) as writer:
        writer.write(mol)
    expected = Chem.SDMolSupplier(str(pose), removeHs=True)[0].GetConformer().GetPositions()

    def forbidden(*args, **kwargs):
        raise AssertionError("Supplied pose must not be re-embedded")

    monkeypatch.setattr(AllChem, "EmbedMolecule", forbidden)
    seen = {}

    class Preparation:
        def prepare(self, prepared):
            seen["coords"] = prepared.GetConformer().GetPositions()[:3]

        def write_pdbqt_file(self, path):
            from pathlib import Path

            Path(path).write_text("REMARK prepared\n", encoding="utf-8")

    def vina(command, **kwargs):
        assert "--score_only" in command and "--out" not in command
        return SimpleNamespace(
            returncode=0, stdout="Estimated Free Energy of Binding : -7.123 (kcal/mol)\n", stderr=""
        )

    monkeypatch.setattr("subprocess.run", vina)
    job = {
        "pose_sdf": str(pose),
        "vina_binary": "vina",
        "receptor_pdbqt": "receptor.pdbqt",
        "pose_mode": "score_only",
    }
    _, score = prepare_and_score_pose(
        job,
        SimpleNamespace(MoleculePreparation=Preparation),
        Chem.RemoveHs(mol),
        tmp_path,
        tmp_path / "box.cfg",
    )
    assert score == -7.123
    assert np.array_equal(seen["coords"], expected)


def test_pose_graph_mismatch_is_not_silently_reembedded(tmp_path):
    pose = tmp_path / "pose.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=9)
    with Chem.SDWriter(str(pose)) as writer:
        writer.write(mol)
    with pytest.raises(ValueError, match="identity"):
        prepare_and_score_pose(
            {"pose_sdf": str(pose)}, None, Chem.MolFromSmiles("[CH3:1][NH2:2]"), tmp_path, None
        )


def test_input_pose_can_resolve_an_unspecified_stereocenter(tmp_path):
    source = Chem.MolFromSmiles("[CH3:1][CH:2]([F:3])[Cl:4]")
    mol = Chem.AddHs(source)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    path = tmp_path / "pose.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    pose = load_input_pose({"pose_sdf": str(path)}, source)
    assert "@" in Chem.MolToSmiles(pose)


def root_stereocenter_pose(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("[C@H:1]([CH3:2])([F:3])[Cl:4]"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    path = tmp_path / "root_stereo.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.SetForceV3000(True)
        writer.write(Chem.RemoveHs(mol))
    return path, Chem.SDMolSupplier(str(path), removeHs=True)[0]


def test_sdf_root_stereocenter_matches_its_own_serialized_request(tmp_path):
    path, pose = root_stereocenter_pose(tmp_path)
    source = Chem.MolFromSmiles(pose_graph_smiles(pose))
    original_bytes = path.read_bytes()
    coordinates = pose.GetConformer().GetPositions().copy()
    assert source.GetAtomWithIdx(0).GetNumExplicitHs() == 1
    assert pose.GetAtomWithIdx(0).GetNumImplicitHs() == 1

    loaded = load_input_pose({"pose_sdf": str(path)}, source)

    assert np.array_equal(loaded.GetConformer().GetPositions(), coordinates)
    assert path.read_bytes() == original_bytes


def test_opposite_specified_root_stereochemistry_still_fails(tmp_path):
    path, pose = root_stereocenter_pose(tmp_path)
    source = Chem.MolFromSmiles(pose_graph_smiles(pose))
    source.GetAtomWithIdx(0).InvertChirality()
    with pytest.raises(ValueError, match="stereochemistry"):
        load_input_pose({"pose_sdf": str(path)}, source)


def test_changed_mapped_connectivity_still_fails(tmp_path):
    path, _ = root_stereocenter_pose(tmp_path)
    source = Chem.MolFromSmiles("[CH3:1][CH:2]([F:3])[Cl:4]")
    with pytest.raises(ValueError, match="connectivity"):
        load_input_pose({"pose_sdf": str(path)}, source)


def test_meeko_identity_uses_the_actual_root_stereo(tmp_path):
    from test_binding_alignment import atom_line

    from agenticgen.workers.plip_alignment import pose_atoms

    _, pose = root_stereocenter_pose(tmp_path)
    metadata = pose_graph_smiles(pose)
    template = Chem.MolFromSmiles(metadata)
    serials = {a.GetAtomMapNum(): a.GetIdx() + 1 for a in pose.GetAtoms()}
    pairs = " ".join(
        f"{a.GetIdx() + 1} {serials[a.GetAtomMapNum()]}" for a in template.GetAtoms()
    )
    lines = ["REMARK SMILES " + metadata, "REMARK SMILES IDX " + pairs]
    lines.extend(
        atom_line(a.GetIdx() + 1, a.GetSymbol(), a.GetIdx(), a.GetSymbol())
        for a in pose.GetAtoms()
    )
    assert {a["map"] for a in pose_atoms("\n".join(lines), pose)} == set(serials)


def test_non_tetrahedral_request_preserves_heavy_atom_maps_and_stereo():
    from agenticgen.workers.plip_alignment import pose_request_smiles

    pose = Chem.MolFromSmiles("[S@SP2H:1]([F:2])([Cl:3])[Br:4]")
    original = Chem.MolToSmiles(pose)
    request = pose_request_smiles(pose)
    parsed = Chem.MolFromSmiles(request)
    assert parsed.GetNumAtoms() == pose.GetNumAtoms() == 4
    assert {a.GetAtomMapNum() for a in parsed.GetAtoms()} == {1, 2, 3, 4}
    assert request == original == pose_request_smiles(parsed)
    assert Chem.MolToSmiles(pose) == original


def test_request_serializer_still_preserves_sdf_root_stereo(tmp_path):
    from agenticgen.workers.plip_alignment import pose_request_smiles

    path, pose = root_stereocenter_pose(tmp_path)
    request = Chem.MolFromSmiles(pose_request_smiles(pose))
    assert load_input_pose({"pose_sdf": str(path)}, request).GetNumAtoms() == 4
    request.GetAtomWithIdx(0).InvertChirality()
    with pytest.raises(ValueError, match="stereochemistry"):
        load_input_pose({"pose_sdf": str(path)}, request)
