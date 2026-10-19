"""Measure pose controls on frozen CK2 inputs without generating or editing molecules."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rdkit import Chem

from agenticgen.adapters.geometry import SynCraftGeometry
from agenticgen.base_config import load_foundation
from agenticgen.chemistry import mapped_smiles, molecule
from agenticgen.contracts import BindingResult
from agenticgen.dataset import load_dataset
from agenticgen.io import read_json, sha256, utc_now, write_json
from agenticgen.selection import contact_retention


def mapped_raw_pose(raw: Chem.Mol, source_smiles: str):
    """Assign a full graph isomorphism; preserve every input heavy-atom coordinate."""
    source = molecule(mapped_smiles(source_smiles))
    pose = Chem.RemoveHs(Chem.Mol(raw))
    matches = pose.GetSubstructMatches(source, useChirality=True, uniquify=False, maxMatches=100001)
    if not matches or len(matches) > 100000 or pose.GetNumAtoms() != source.GetNumAtoms():
        raise ValueError("No complete, enumerable source-to-pose graph correspondence")
    match = min(matches)
    for atom in pose.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom, raw_index in zip(source.GetAtoms(), match, strict=True):
        pose.GetAtomWithIdx(raw_index).SetAtomMapNum(atom.GetAtomMapNum())
    return pose, {"isomorphism_count": len(matches), "selected_source_to_raw_indices": list(match)}


def pose_distance(raw, docked):
    """Symmetry-corrected displacement in the common pocket frame, without fitting."""
    # The frozen SMILES are checked separately. SDF readers may assign extra
    # chiral labels from 3D; those must not prevent displacement measurement.
    matches = raw.GetSubstructMatches(docked, useChirality=False, uniquify=False, maxMatches=100000)
    a = raw.GetConformer().GetPositions()
    b = docked.GetConformer().GetPositions()
    if not matches or len(a) != len(b):
        raise ValueError("Pose RMSD requires the same entire molecular graph")
    rmsds = [float(np.sqrt(np.mean(np.sum((a[list(m)] - b) ** 2, axis=1)))) for m in matches]
    return {
        "minimum_symmetry_rmsd_no_alignment_angstrom": min(rmsds),
        "centroid_shift_angstrom": float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))),
        "symmetry_assignment_count": len(matches),
    }


def compare(first, second, permutations=None):
    result = contact_retention(first, second)
    if first.status != "ok" or second.status != "ok":
        return result
    directional = {"HydrogenBond", "SaltBridge", "PiStacking", "PiCation", "HalogenBond"}
    a = {
        (i.kind, i.residue, tuple(sorted(i.atom_maps)))
        for i in first.interactions
        if i.kind in directional
    }
    b = {
        (i.kind, i.residue, tuple(sorted(i.atom_maps)))
        for i in second.interactions
        if i.kind in directional
    }
    result.update(
        directional_source_count=len(a),
        directional_retained_count=len(a & b),
        directional_exact_recall=len(a & b) / len(a) if a else None,
    )
    if permutations and result.get("source_contact_count"):
        target = {(i.kind, i.residue, tuple(sorted(i.atom_maps))) for i in second.interactions}
        recalls = []
        for permutation in permutations:
            original = {
                (i.kind, i.residue, tuple(sorted(permutation.get(m, m) for m in i.atom_maps)))
                for i in first.interactions
            }
            recalls.append(len(original & target) / len(original))
        result["symmetry_equivalent_exact_recall_min"] = min(recalls)
        result["symmetry_equivalent_exact_recall_max"] = max(recalls)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    master = read_json(run / "manifest.json")
    config = load_foundation(Path(master["config"]))
    dataset = load_dataset(config.resolve(config.value.dataset))
    supplier = Chem.SDMolSupplier(
        str(config.resolve(config.value.dataset).parent / dataset.sdf_file), removeHs=False
    )
    output.mkdir(parents=True, exist_ok=args.resume)
    evaluator = SynCraftGeometry(config)
    write_json(
        output / "manifest.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "source_run": str(run),
            "source_manifest_sha256": sha256(run / "manifest.json"),
            "workers": args.workers,
            "new_api_calls": 0,
            "geometry_protocol": evaluator.protocol,
            "modes": ["raw_score_only", "raw_local_only", "redocked_score_only"],
            "raw_heavy_atom_embedding_or_minimization": False,
        },
    )

    def execute(case):
        folder = output / "cases" / case.case_id
        folder.mkdir(parents=True, exist_ok=args.resume)
        original_result = read_json(
            run / "runs" / case.case_id / "cases" / case.case_id / "result.json"
        )
        raw, correspondence = mapped_raw_pose(supplier[case.sdf_record_index], case.smiles)
        raw_path = folder / "raw_mapped.sdf"
        if not raw_path.exists():
            with Chem.SDWriter(str(raw_path)) as writer:
                writer.SetForceV3000(True)
                writer.write(raw)
        write_json(folder / "correspondence.json", correspondence)
        redock_path = (
            run
            / "runs"
            / case.case_id
            / "cases"
            / case.case_id
            / "geometry_original/docked_pose.sdf"
        )
        docked = Chem.SDMolSupplier(str(redock_path), removeHs=False)[0]
        baseline = BindingResult.model_validate(original_result["source_binding"])

        def measured_pose(pose_path, target, mode):
            response = target / "response.json"
            if args.resume and response.exists():
                job = read_json(target / "request.json")
                if job["input_pose_sha256"] != sha256(pose_path) or job["pose_mode"] != mode:
                    raise ValueError("Existing measurement input differs from control")
                return BindingResult.model_validate(read_json(response))
            return evaluator.evaluate_pose(mapped_smiles(case.smiles), pose_path, target, mode=mode)

        raw_binding = measured_pose(raw_path, folder / "raw_score_only", "score_only")
        local_binding = measured_pose(raw_path, folder / "raw_local_only", "local_only")
        repeat_binding = measured_pose(redock_path, folder / "redocked_score_only", "score_only")
        permutations = []
        for match in raw.GetSubstructMatches(
            raw, useChirality=True, uniquify=False, maxMatches=100000
        ):
            permutations.append(
                {
                    raw.GetAtomWithIdx(i).GetAtomMapNum(): raw.GetAtomWithIdx(j).GetAtomMapNum()
                    for i, j in enumerate(match)
                }
            )
        candidates = original_result["candidates"]
        product = next((c for c in candidates if c.get("binding", {}).get("status") == "ok"), None)
        result = {
            "case_id": case.case_id,
            "rank": case.selection_rank,
            "raw_input_sha256": sha256(raw_path),
            "raw_pose_correspondence": correspondence,
            "pose_displacement": pose_distance(raw, docked),
            "raw_score_only": raw_binding.model_dump(mode="json"),
            "raw_local_only": local_binding.model_dump(mode="json"),
            "saved_redock": baseline.model_dump(mode="json"),
            "redocked_score_only": repeat_binding.model_dump(mode="json"),
            "raw_to_redock": compare(raw_binding, baseline, permutations),
            "raw_to_local": compare(raw_binding, local_binding, permutations),
            "same_redocked_pose_repeat": compare(baseline, repeat_binding),
            "redock_to_product": product.get("contact_retention") if product else None,
            "raw_to_product": compare(
                raw_binding, BindingResult.model_validate(product["binding"]), permutations
            )
            if product
            else None,
        }
        if local_binding.status == "ok":
            local_pose = Chem.SDMolSupplier(
                str(folder / "raw_local_only/docked_pose.sdf"), removeHs=False
            )[0]
            result["local_pose_displacement"] = pose_distance(raw, local_pose)
        write_json(folder / "result.json", result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute, case) for case in dataset.cases]):
            result = future.result()
            results.append(result)
            print({"finished": len(results), "case_id": result["case_id"]}, flush=True)
    results.sort(key=lambda r: r["rank"])
    rows = []
    for r in results:
        row = {
            "rank": r["rank"],
            "case_id": r["case_id"],
            "rmsd_no_alignment": r["pose_displacement"][
                "minimum_symmetry_rmsd_no_alignment_angstrom"
            ],
            "centroid_shift": r["pose_displacement"]["centroid_shift_angstrom"],
            "mapping_count": r["raw_pose_correspondence"]["isomorphism_count"],
        }
        for mode in ["raw_score_only", "raw_local_only", "saved_redock", "redocked_score_only"]:
            row[mode + "_status"] = r[mode]["status"]
            row[mode + "_vina"] = r[mode]["score_kcal_mol"]
        for mode in [
            "raw_to_redock",
            "raw_to_local",
            "same_redocked_pose_repeat",
            "redock_to_product",
            "raw_to_product",
        ]:
            for metric in [
                "fraction",
                "paper_recall",
                "paper_jaccard",
                "directional_exact_recall",
                "symmetry_equivalent_exact_recall_max",
            ]:
                row[mode + "_" + metric] = (r[mode] or {}).get(metric)
        rows.append(row)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = {}
    for key in rows[0]:
        values = [
            r[key] for r in rows if isinstance(r[key], (float, int)) and math.isfinite(r[key])
        ]
        if values:
            stats[key] = {
                "n": len(values),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "min": min(values),
                "max": max(values),
            }
    write_json(
        output / "summary.json",
        {"case_count": len(results), "new_api_calls": 0, "statistics": stats},
    )
    manifest = read_json(output / "manifest.json")
    manifest.update(status="completed", ended_at=utc_now(), case_count=len(results))
    write_json(output / "manifest.json", manifest)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
