"""Small shared molecular identity functions; coordinates are never copied onto an edit."""

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from agenticgen.io import object_hash


def molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty molecule")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("A connected molecule is required")
    return mol


def canonical_smiles(mol_or_smiles: Chem.Mol | str, *, maps: bool = False) -> str:
    mol = molecule(mol_or_smiles) if isinstance(mol_or_smiles, str) else Chem.Mol(mol_or_smiles)
    if not maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        # Atom-map labels distinguish otherwise equivalent substituents. Recompute
        # stereo after removing them (e.g. two equivalent sulfonyl oxygens).
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def mapped_smiles(smiles: str) -> str:
    """Use the same input-order 1-based mapping as SynCraft.create_user_prompt."""
    mol = molecule(smiles)
    if mol.GetNumAtoms() >= 500:
        raise ValueError("Source atom IDs would collide with SynCraft's new-atom namespace")
    for index, atom in enumerate(mol.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    return canonical_smiles(mol, maps=True)


def graph_hash(smiles: str) -> str:
    return object_hash(
        {
            "smiles": canonical_smiles(smiles),
            "rdkit": rdBase.rdkitVersion,
            "normalization": "canonical-isomeric-clean-stereo-after-unmapping-v2",
        }
    )


def similarity(left: str, right: str, bits: int = 2048) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(molecule(left)), generator.GetFingerprint(molecule(right))
        )
    )
