from rdkit import Chem

from agenticgen.adapters.geometry import response_identity_matches
from agenticgen.chemistry import canonical_smiles


def test_fused_aromatic_response_is_compared_after_equal_parsing():
    mol = Chem.MolFromSmiles(
        "[cH:2]1[cH:3][c:4]2[cH:5][c:6]([cH:7]1)[CH:1045]2[C:1032]12"
        "[c:1031]3[c:15]([cH:1037][n:13][cH:1039][cH:1046]3)[C:1042]([CH3:8])"
        "=[CH:1038][C:1025](=[O:22])[C:19]([CH:1044]([OH:21])[NH2:1041])"
        "=[C:1034]1[NH:1048]2"
    )
    response = canonical_smiles(mol)
    # RDKit 2026.03 changes the aromatic spelling on reparse; newer releases
    # can already be idempotent. Both must accept this same-graph response.
    assert response_identity_matches(mol, response)
    assert not response_identity_matches(mol, response.replace("C(N)O", "C(N)F"))


def test_response_comparison_rejects_opposite_stereo_and_charge():
    source = Chem.MolFromSmiles("[C@H:1]([CH3:2])([F:3])[Cl:4]")
    assert response_identity_matches(source, canonical_smiles(source))
    assert response_identity_matches(source, "C[C@@H](F)Cl")
    assert not response_identity_matches(source, "C[C@H](F)Cl")
    charged = Chem.MolFromSmiles("[NH3+:1][CH3:2]")
    assert not response_identity_matches(charged, "CN")
