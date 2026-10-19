"""Map Meeko/Vina atom identities to PLIP contacts without inferring bond orders from coordinates.

This worker helper deliberately has no controller-package imports.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


def pose_atoms(text, source):
    from rdkit import Chem

    lines = text.splitlines()
    if any(line.startswith("MODEL") for line in lines):
        begin = next(i for i, line in enumerate(lines) if line.startswith("MODEL"))
        end = next(i for i in range(begin + 1, len(lines)) if lines[i].startswith("ENDMDL"))
        # Some Vina versions keep metadata outside MODEL.
        lines = [line for line in lines[:begin] if line.startswith("REMARK")] + lines[begin:end]
    smiles = [
        line[len("REMARK SMILES ") :].strip()
        for line in lines
        if line.startswith("REMARK SMILES ") and not line.startswith("REMARK SMILES IDX")
    ]
    if len(smiles) != 1:
        raise ValueError("Docked pose must retain exactly one Meeko SMILES identity")
    template = Chem.MolFromSmiles(smiles[0])
    if template is None:
        raise ValueError("Invalid pose SMILES metadata")

    def identity(mol):
        copy = Chem.Mol(mol)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
        return Chem.MolToSmiles(copy, isomericSmiles=True)

    if identity(template) != identity(source):
        raise ValueError("Docking preparation changed source identity/stereochemistry")
    if Chem.MolToSmiles(template, isomericSmiles=True) != Chem.MolToSmiles(
        source, isomericSmiles=True
    ):
        raise ValueError("Docking preparation changed source atom-map connectivity")
    source_atoms = {a.GetAtomMapNum(): a for a in source.GetAtoms() if a.GetAtomicNum() > 1}
    metadata = []
    for line in lines:
        if line.startswith("REMARK SMILES IDX "):
            metadata.extend(int(i) for i in line[len("REMARK SMILES IDX ") :].split())
    if len(metadata) % 2 or not metadata:
        raise ValueError("Docked pose is missing Meeko SMILES IDX metadata")
    serial_to_map = {}
    for atom_index, serial in zip(metadata[::2], metadata[1::2], strict=True):
        atom = template.GetAtomWithIdx(atom_index - 1)
        if atom.GetAtomicNum() > 1:
            atom_map = atom.GetAtomMapNum()
            if atom_map not in source_atoms or serial in serial_to_map:
                raise ValueError("Pose atom-map metadata differs from source")
            if atom.GetAtomicNum() != source_atoms[atom_map].GetAtomicNum():
                raise ValueError("Pose atom element differs from source")
            serial_to_map[serial] = atom_map
    if set(serial_to_map.values()) != set(source_atoms) or len(serial_to_map) != len(source_atoms):
        raise ValueError("Pose omits or duplicates a source heavy atom")
    atoms = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        serial = int(line[6:11])
        coords = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        if not all(math.isfinite(v) for v in coords):
            raise ValueError("Nonfinite docking coordinates")
        atom_map = serial_to_map.get(serial)
        element = source_atoms[atom_map].GetSymbol() if atom_map is not None else "H"
        if atom_map is None and not line.split()[-1].startswith("H"):
            raise ValueError("Unmapped heavy atom in docking output")
        atoms.append({"serial": serial, "map": atom_map, "element": element, "coords": coords})
    if len({a["serial"] for a in atoms}) != len(atoms):
        raise ValueError("Duplicate docked atom serial")
    if {a["serial"] for a in atoms if a["map"] is not None} != set(serial_to_map):
        raise ValueError("Pose coordinates and Meeko mapping disagree")
    return atoms


def write_complex(receptor: Path, atoms, path: Path):
    lines = [
        line
        for line in receptor.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    if not lines:
        raise ValueError("Receptor PDB has no atoms")
    offset = max(int(line[6:11]) for line in lines)
    used_chains = {line[21:22] for line in lines}
    chain = next((c for c in "ZYXWVUTSRQPONMLKJIHGFEDCBA" if c not in used_chains), None)
    if chain is None or offset + len(atoms) >= 100000:
        raise ValueError("No unambiguous PDB ligand identity/serial range")
    mapping = {}
    for index, atom in enumerate(atoms, 1):
        serial = offset + index
        x, y, z = atom["coords"]
        element = atom["element"]
        name = (element + str(index))[:4]
        lines.append(
            f"HETATM{serial:5d} {name:<4s} LIG {chain}   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{0:6.2f}          {element:>2s}  "
        )
        if atom["map"] is not None:
            mapping[serial] = atom["map"]
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="utf-8")
    return mapping, chain


def plip_serial_mapping(path: Path, atoms, ligand_chain: str):
    """Follow PLIP PDB renumbering using preserved atom names AND unchanged coordinates."""
    expected = {}
    for index, atom in enumerate(atoms, 1):
        if atom["map"] is not None:
            name = (atom["element"] + str(index))[:4]
            if name in expected:
                raise ValueError("Ambiguous ligand atom names in PDB")
            expected[name] = atom
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("HETATM") or line[17:20] != "LIG" or line[21:22] != ligand_chain:
            continue
        name = line[12:16].strip()
        if name not in expected:
            # PLIP may add hydrogens. Unknown heavy atoms cannot be silently ignored.
            if line[76:78].strip() not in {"H", "D"}:
                raise ValueError("PLIP introduced an unknown ligand heavy atom")
            continue
        atom = expected[name]
        coords = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        if line[76:78].strip() != atom["element"] or any(
            abs(a - b) > 0.002 for a, b in zip(coords, atom["coords"], strict=True)
        ):
            raise ValueError("PLIP atom-name/coordinate/element mapping is inconsistent")
        serial = int(line[6:11])
        if serial in result or atom["map"] in result.values():
            raise ValueError("Duplicate PLIP ligand identity")
        result[serial] = atom["map"]
    if set(result.values()) != {a["map"] for a in expected.values()}:
        raise ValueError("PLIP ligand mapping is incomplete")
    return result


def contacts_from_xml(path: Path, serial_to_map: dict[int, int], ligand_chain: str):
    root = ET.parse(path).getroot()
    definitions = {
        "hydrophobic_interactions/hydrophobic_interaction": ("Hydrophobic", ("ligcarbonidx",)),
        "hydrogen_bonds/hydrogen_bond": ("HydrogenBond", ("acceptoridx", "donoridx")),
        "salt_bridges/salt_bridge": ("SaltBridge", ("lig_idx_list/idx",)),
        "pi_stacks/pi_stack": ("PiStacking", ("lig_idx_list/idx", "lig_ring_atoms_idx/idx")),
        "pi_cation_interactions/pi_cation_interaction": (
            "PiCation",
            ("lig_idx_list/idx", "lig_ring_atoms_idx/idx"),
        ),
        "halogen_bonds/halogen_bond": ("HalogenBond", ("don_idx",)),
        "water_bridges/water_bridge": ("WaterBridge", ("donor_idx", "acceptor_idx", "ligatom_idx")),
        "metal_complexes/metal_complex": (
            "MetalComplex",
            ("target_idx", "metal_idx", "ligatom_idx"),
        ),
    }
    contacts = {}
    sites = [
        site
        for site in root.findall(".//bindingsite")
        if site.findtext("identifiers/hetid") == "LIG"
        and site.findtext("identifiers/chain") == ligand_chain
    ]
    if len(sites) != 1:
        raise ValueError("PLIP report does not identify exactly one requested ligand")
    for query, (kind, tags) in definitions.items():
        for item in sites[0].findall("interactions/" + query):
            ids = {int(node.text) for tag in tags for node in item.findall(tag) if node.text}
            if (
                kind in {"PiStacking", "PiCation", "SaltBridge", "Hydrophobic", "HalogenBond"}
                and not ids <= serial_to_map.keys()
            ):
                raise ValueError("PLIP ligand indices are not completely mapped")
            if kind == "HydrogenBond" and item.findtext("protisdon") is not None:
                ligand_tag = (
                    "acceptoridx" if item.findtext("protisdon").lower() == "true" else "donoridx"
                )
                ids = {int(item.findtext(ligand_tag))}
                if not ids <= serial_to_map.keys():
                    raise ValueError("PLIP hydrogen-bond ligand atom is not mapped")
            maps = sorted({serial_to_map[i] for i in ids if i in serial_to_map})
            if not maps:
                continue
            residue = (
                f"{item.findtext('restype')}{item.findtext('resnr')}.{item.findtext('reschain')}"
            )
            contacts[(kind, residue, tuple(maps))] = {
                "kind": kind,
                "residue": residue,
                "atom_maps": maps,
            }
    return list(contacts.values())
