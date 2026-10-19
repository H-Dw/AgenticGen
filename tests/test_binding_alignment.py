import pytest
from rdkit import Chem

from agenticgen.workers.plip_alignment import (
    contacts_from_xml,
    plip_serial_mapping,
    pose_atoms,
    write_complex,
)


def atom_line(serial, name, x, element):
    return (
        f"ATOM  {serial:5d} {name:<4s} UNL     1    "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{0.0:6.2f}     0.000 {element}"
    )


def test_meeko_metadata_controls_maps_despite_pose_order_and_receptor_end(tmp_path):
    source = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    text = "\n".join(
        [
            "MODEL 1",
            "REMARK SMILES [CH3:1][CH2:2][OH:3]",
            "REMARK SMILES IDX 1 8 2 4 3 31",
            atom_line(31, "O", 3, "OA"),
            atom_line(8, "C", 1, "C"),
            atom_line(4, "C", 2, "C"),
            "ENDMDL",
        ]
    )
    atoms = pose_atoms(text, source)
    assert [a["map"] for a in atoms] == [3, 1, 2]
    receptor = tmp_path / "protein.pdb"
    receptor.write_text(atom_line(101, "C", 0, "C") + "\nEND\n", encoding="utf-8")
    mapping, chain = write_complex(receptor, atoms, tmp_path / "complex.pdb")
    assert mapping == {102: 3, 103: 1, 104: 2}
    assert chain == "Z"
    complex_path = tmp_path / "complex.pdb"
    corrected = []
    for line in complex_path.read_text().splitlines():
        if line.startswith("HETATM"):
            line = line[:6] + f"{int(line[6:11]) - 13:5d}" + line[11:]
        corrected.append(line)
    fixed = tmp_path / "plipfixed.pdb"
    fixed.write_text("\n".join(corrected) + "\n", encoding="utf-8")
    assert plip_serial_mapping(fixed, atoms, chain) == {89: 3, 90: 1, 91: 2}
    with pytest.raises(ValueError, match="missing Meeko"):
        pose_atoms(text.replace("REMARK SMILES IDX 1 8 2 4 3 31", ""), source)
    with pytest.raises(ValueError, match="identity"):
        pose_atoms(text.replace("[OH:3]", "[NH2:3]"), source)


def test_plip_selects_requested_ligand_and_only_ligand_side_indices(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text(
        """<report><bindingsite><identifiers><hetid>LIG</hetid><chain>Z</chain></identifiers>
    <interactions><hydrogen_bonds><hydrogen_bond><restype>LYS</restype><resnr>68</resnr>
    <reschain>A</reschain><acceptoridx>101</acceptoridx><donoridx>104</donoridx>
    </hydrogen_bond></hydrogen_bonds><pi_stacks><pi_stack><restype>PHE</restype><resnr>113</resnr>
    <reschain>A</reschain><lig_idx_list><idx>102</idx><idx>103</idx></lig_idx_list></pi_stack></pi_stacks>
    </interactions></bindingsite><bindingsite><identifiers><hetid>OTHER</hetid><chain>A</chain>
    </identifiers><interactions/></bindingsite></report>""",
        encoding="utf-8",
    )
    records = contacts_from_xml(path, {102: 500, 103: 2, 104: 3}, "Z")
    assert {r["kind"]: r["atom_maps"] for r in records} == {
        "HydrogenBond": [3],
        "PiStacking": [2, 500],
    }
    with pytest.raises(ValueError, match="exactly one"):
        contacts_from_xml(path, {102: 1}, "Y")
