import numpy as np
from rdkit import Chem

from scripts.audit_ck2_pose_controls import mapped_raw_pose, pose_distance


def positioned(smiles, coordinates):
    mol = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for index, point in enumerate(coordinates):
        conformer.SetAtomPosition(index, point)
    mol.AddConformer(conformer)
    return mol


def test_raw_pose_mapping_changes_no_coordinates_and_handles_input_order():
    raw = positioned("CCO", [(0, 0, 0), (1, 0, 0), (2, 1, 0)])
    mapped, receipt = mapped_raw_pose(raw, "OCC")
    np.testing.assert_array_equal(
        raw.GetConformer().GetPositions(), mapped.GetConformer().GetPositions()
    )
    assert [a.GetAtomMapNum() for a in mapped.GetAtoms()] == [3, 2, 1]
    assert receipt["isomorphism_count"] == 1


def test_pose_displacement_does_not_superimpose_away_a_binding_orientation_change():
    raw = positioned("CCO", [(0, 0, 0), (1, 0, 0), (2, 1, 0)])
    rotated = positioned("CCO", [(0, 0, 0), (-1, 0, 0), (-2, -1, 0)])
    assert pose_distance(raw, rotated)["minimum_symmetry_rmsd_no_alignment_angstrom"] > 2


def test_extra_3d_stereo_labels_do_not_prevent_graph_displacement_measurement():
    raw = positioned("CC(O)F", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)])
    docked = Chem.Mol(raw)
    docked.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    assert pose_distance(raw, docked)["minimum_symmetry_rmsd_no_alignment_angstrom"] == 0
