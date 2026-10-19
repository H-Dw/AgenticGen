from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from agenticgen.workers.syncraft_geometry import prepare_and_score_pose


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
