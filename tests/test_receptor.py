import subprocess
from types import SimpleNamespace

import pytest
from rdkit import Chem

from agenticgen.receptor import prepare_receptor


def test_reference_ligand_box_prevents_whole_protein_centroid_fallback(tmp_path, monkeypatch):
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        "ATOM      1  CA  ALA A   1      99.000  99.000  99.000\n"
        "HETATM    2  C   OLD A   2      99.000  99.000  99.000\nEND\n",
        encoding="utf-8",
    )
    molecule = Chem.MolFromSmiles("CCO")
    conf = Chem.Conformer(3)
    for i, point in enumerate([(0, 0, 0), (1, 2, 3), (4, 6, 8)]):
        conf.SetAtomPosition(i, point)
    molecule.AddConformer(conf)
    ligand = tmp_path / "reference.sdf"
    with Chem.SDWriter(str(ligand)) as writer:
        writer.write(molecule)
    output = tmp_path / "prepared"

    def fake_docker(command, **kwargs):
        if command[1:3] == ["image", "inspect"]:
            return SimpleNamespace(stdout="sha256:fixture\n", returncode=0)
        assert "none" in command
        (output / "receptor.pdbqt").write_text("ATOM fixture\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_docker)
    result = prepare_receptor(protein, ligand, output, "fixture")
    assert result["status"] == "prepared"
    assert result["center_A"] == [2, 3, 4]
    assert result["size_A"] == [22.5, 22.5, 24]
    assert "HETATM" not in (output / "receptor.pdb").read_text()
    with pytest.raises(FileExistsError):
        prepare_receptor(protein, ligand, output, "fixture")
