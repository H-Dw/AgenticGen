"""Standalone upstream docking/PLIP worker; no AgenticGen install required."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def prepare_and_score_pose(job, upstream, source, out, box):
    """Preserve a supplied pose for score-only/local-only or seeded redocking."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(job["pose_sdf"], removeHs=False)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError("Pose evaluation requires exactly one readable SDF record")
    pose = Chem.RemoveHs(supplier[0])
    if pose.GetNumConformers() != 1 or not pose.GetConformer().Is3D():
        raise ValueError("Input pose has no three-dimensional conformer")
    if Chem.MolToSmiles(pose, isomericSmiles=True) != Chem.MolToSmiles(source, isomericSmiles=True):
        raise ValueError("Input pose graph or atom-map identity differs from requested molecule")
    # Add only hydrogen coordinates. Never embed or minimize the supplied heavy atoms.
    pose = Chem.AddHs(pose, addCoords=True)
    prepared = out / "input_pose.pdbqt"
    prep = upstream.MoleculePreparation()
    prep.prepare(pose)
    prep.write_pdbqt_file(str(prepared))
    command = [
        job["vina_binary"],
        "--config",
        str(box),
        "--ligand",
        str(prepared),
        "--receptor",
        job["receptor_pdbqt"],
    ]
    mode = job["pose_mode"]
    result_path = prepared
    if mode in {"score_only", "local_only"}:
        command.append("--" + mode)
    if mode != "score_only":
        result_path = out / "evaluated_pose.pdbqt"
        command += ["--out", str(result_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (out / "vina_pose.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("Vina pose evaluation failed; inspect vina_pose.log")
    score_text = completed.stdout + "\n" + result_path.read_text(encoding="utf-8")
    score = re.search(
        r"(?:Affinity:\s*|REMARK VINA RESULT:\s*|Estimated Free Energy of Binding\s*:\s*)([-+\d.]+)",
        score_text,
    )
    if score is None:
        raise ValueError("Vina pose evaluation returned no score")
    return result_path, float(score.group(1))


def run_job(job):
    from plip_alignment import contacts_from_xml, plip_serial_mapping, pose_atoms, write_complex
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    out = Path(job["output"])
    os.chdir(out)
    os.environ["SYNCRAFT_VINA_BIN"] = job["vina_binary"]
    os.environ["OMP_NUM_THREADS"] = str(job["cpu_threads"])
    sys.path.insert(0, str(Path(job["syncraft_root"]) / "src"))
    import extract_interaction as upstream

    source = Chem.MolFromSmiles(job["smiles"])
    if source is None:
        raise ValueError("Invalid source graph")
    config = Path(job["vina_config"]).read_text(encoding="utf-8")
    controlled = [
        line
        for line in config.splitlines()
        if line.split("=", 1)[0].strip() not in {"seed", "cpu", "exhaustiveness"}
    ]
    controlled += [
        f"seed = {job['seed']}",
        f"cpu = {job['cpu_threads']}",
        f"exhaustiveness = {job['exhaustiveness']}",
    ]
    box = out / "effective_vina.cfg"
    box.write_text("\n".join(controlled) + "\n", encoding="utf-8")
    rdBase.SeedRandomNumberGenerator(job["seed"])
    # Fix the seed of upstream's existing RDKit embedding call; all other ligand
    # preparation and docking code executes from extract_interaction.py unchanged.
    embed = AllChem.EmbedMolecule

    def seeded_embed(mol, *args, **kwargs):
        kwargs.setdefault("randomSeed", job["seed"])
        return embed(mol, *args, **kwargs)

    AllChem.EmbedMolecule = seeded_embed
    if job.get("pose_sdf"):
        docked, score_value = prepare_and_score_pose(job, upstream, source, out, box)
    else:
        docked = upstream.run_docking(
            job["smiles"], job["receptor_pdbqt"], str(box), str(out / "docking")
        )
    if not docked:
        raise RuntimeError("Upstream Vina failed")
    text = Path(docked).read_text(encoding="utf-8")
    if not job.get("pose_sdf"):
        score = re.search(r"REMARK VINA RESULT:\s*([-+\d.]+)", text)
        if not score:
            raise ValueError("Vina output has no score")
        score_value = float(score.group(1))
    # Vina retains Meeko's SMILES IDX metadata. Use that explicit bijection
    # instead of guessing atom identity from the output line order or geometry.
    atoms = pose_atoms(text, source)
    complex_path = out / "complex.pdb"
    serial_to_map, chain = write_complex(Path(job["receptor_pdb"]), atoms, complex_path)
    plip_out = out / "plip"
    subprocess.run(["plip", "-f", str(complex_path), "-x", "-o", str(plip_out)], check=True)
    reports = list(plip_out.glob("*.xml"))
    if len(reports) != 1:
        raise ValueError("Expected exactly one PLIP XML report for the requested complex")
    fixed = list(plip_out.glob("plipfixed.*.pdb"))
    if len(fixed) > 1:
        raise ValueError("PLIP produced ambiguous corrected coordinate files")
    corrected_mapping = plip_serial_mapping(fixed[0] if fixed else complex_path, atoms, chain)
    contacts = contacts_from_xml(reports[0], corrected_mapping, chain)
    original = Chem.RemoveHs(source)

    def plain(m):
        copy = Chem.Mol(m)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(copy, isomericSmiles=True)

    upstream_contacts = [
        {
            "type": c["kind"],
            "residue": c["residue"],
            "ligand_atom_indices": [i - 1 for i in c["atom_maps"]],
        }
        for c in contacts
    ]
    conf = Chem.Conformer(original.GetNumAtoms())
    indices = {a.GetAtomMapNum(): a.GetIdx() for a in original.GetAtoms()}
    for atom in atoms:
        if atom["map"] is not None:
            conf.SetAtomPosition(indices[atom["map"]], atom["coords"])
    original.RemoveAllConformers()
    original.AddConformer(conf)
    with Chem.SDWriter(str(out / "docked_pose.sdf")) as writer:
        writer.write(original)
    (out / "atom_alignment.json").write_text(
        json.dumps(
            {
                "complex_serial_to_source_map": serial_to_map,
                "plip_serial_to_source_map": corrected_mapping,
                "ligand_chain": chain,
                "selection": "Meeko SMILES IDX, then PLIP names/element/coordinate bijection; no symmetry rematching",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "protocol_id": job["protocol_id"],
        "graph_smiles": plain(source),
        "score_kcal_mol": score_value,
        "evaluation_mode": job.get("pose_mode", "redock"),
        "input_pose_sha256": job.get("input_pose_sha256"),
        "interactions": contacts,
        "prompt": upstream.format_interactions_for_llm(upstream_contacts, job["smiles"]),
        "artifact_paths": [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()],
    }


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run_job(request)
    Path(request["output"], "response.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
