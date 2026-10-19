from rdkit import Chem
from rdkit.Chem import AllChem
from test_binding_alignment import atom_line

from agenticgen.chemistry import canonical_smiles, graph_hash
from agenticgen.workers.plip_alignment import pose_atoms


def test_atom_maps_do_not_create_chemical_chirality():
    mapped = "[NH2:21][S@:1041](=[O:22])(=[O:1048])[CH3:1040]"
    assert canonical_smiles(mapped) == canonical_smiles("CS(N)(=O)=O")
    assert graph_hash(mapped) == graph_hash("CS(N)(=O)=O")
    assert canonical_smiles("[F:1][C@H:2]([Cl:3])[Br:4]") != canonical_smiles(
        "[F:1][C@@H:2]([Cl:3])[Br:4]"
    )


def test_meeko_metadata_sulfone_stereo_is_checked_after_unmapping(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH2:21][S:1041](=[O:22])(=[O:1048])[CH3:1040]"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    path = tmp_path / "sulfone.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.SetForceV3000(True)
        writer.write(mol)
    source = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    metadata = Chem.MolToSmiles(source)
    template = Chem.MolFromSmiles(metadata)
    serials = {a.GetAtomMapNum(): a.GetIdx() + 1 for a in source.GetAtoms()}
    index_pairs = " ".join(
        f"{a.GetIdx() + 1} {serials[a.GetAtomMapNum()]}" for a in template.GetAtoms()
    )
    text = "\n".join(
        [
            "REMARK SMILES " + metadata,
            "REMARK SMILES IDX " + index_pairs,
            *(
                atom_line(a.GetIdx() + 1, a.GetSymbol(), a.GetIdx(), a.GetSymbol())
                for a in source.GetAtoms()
            ),
        ]
    )
    assert {a["map"] for a in pose_atoms(text, source)} == set(serials)
