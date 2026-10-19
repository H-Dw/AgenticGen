"""Select CK2 affinity/quality cohorts before observing any agent outcomes."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from agenticgen.chemistry import canonical_smiles
from agenticgen.contracts import CaseDataset, SourceCase
from agenticgen.dataset import METRICS, load_dataset
from agenticgen.errors import ArtifactError
from agenticgen.io import sha256, utc_now, write_json

Row = dict[str, Any]


def read_csv(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        raise ValueError("Cannot export an empty cohort")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flag(row: Row, name: str) -> bool:
    if row[name] not in (True, False, "True", "False"):
        raise ArtifactError(f"Invalid boolean in {name}")
    return row[name] in (True, "True")


def unique_graphs(rows: list[Row]) -> list[Row]:
    """Choose highest-affinity qualifying pose; break ties by relief then ID."""
    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r["pic50_on_rescore"]),
            float(r["mmff_relief_per_heavy"]),
            r["node_id"],
        ),
    )
    by_graph: dict[str, Row] = {}
    for row in ordered:
        by_graph.setdefault(row["canonical_smiles"], row)
    return list(by_graph.values())


def select_cohorts(rows: list[Row], count_per_stratum: int = 3) -> dict[str, Any]:
    if not 1 <= count_per_stratum <= 5:
        raise ValueError("Select 1 to 5 candidates per stratum")
    valid = []
    for source in rows:
        if flag(source, "valid_connected"):
            row = {**source, "canonical_smiles": canonical_smiles(source["smiles"])}
            if not all(math.isfinite(float(row[k])) for k in METRICS):
                raise ArtifactError("Valid-cohort selection metrics must be finite")
            valid.append(row)
    if not valid or len({r["node_id"] for r in rows}) != len(rows):
        raise ArtifactError("Expected nonempty valid cohort and unique particle IDs")
    affinity = float(np.quantile([float(r["pic50_on_rescore"]) for r in valid], 0.9))
    relief = float(np.quantile([float(r["mmff_relief_per_heavy"]) for r in valid], 0.75))
    high = [r for r in valid if float(r["pic50_on_rescore"]) >= affinity]
    quality = [
        r
        for r in high
        if flag(r, "pb_full_pass")
        and r["energy_status"] == "converged"
        and float(r["mmff_relief_per_heavy"]) <= relief
    ]
    # Catch mismatched source tables or an inadvertent change in the previous gate.
    high_ids, quality_ids = {r["node_id"] for r in high}, {r["node_id"] for r in quality}
    for row in rows:
        if flag(row, "affinity_tail") != (row["node_id"] in high_ids):
            raise ArtifactError("Affinity gate differs from recorded source labels")
        if flag(row, "quality_tail") != (row["node_id"] in quality_ids):
            raise ArtifactError("Quality gate differs from recorded source labels")
    high_unique, quality_unique = unique_graphs(high), unique_graphs(quality)
    quality_graphs = {r["canonical_smiles"] for r in quality_unique}
    # Challenge compounds remain explicitly outside the quality cohort. Select
    # among eligible poses BEFORE deduplication, so a bad pose cannot hide a good one.
    challenge = unique_graphs(
        [
            r
            for r in high
            if r["canonical_smiles"] not in quality_graphs
            and flag(r, "pb_full_pass")
            and r["energy_status"] == "converged"
            and float(r["sa_score"]) >= 3.0
        ]
    )
    selected = []
    for name, cohort in (
        ("quality_relative_high_sa", quality_unique),
        ("high_sa_high_strain_challenge", challenge),
    ):
        ordered = sorted(
            cohort,
            key=lambda r: (-float(r["sa_score"]), -float(r["pic50_on_rescore"]), r["node_id"]),
        )[:count_per_stratum]
        if len(ordered) != count_per_stratum:
            raise ArtifactError(f"Insufficient candidates in {name}; do not silently relax gates")
        selected.extend(
            {**r, "validation_stratum": name, "stratum_rank": i + 1} for i, r in enumerate(ordered)
        )
    return {
        "thresholds": {
            "affinity_valid_particle_q90": affinity,
            "relief_per_heavy_valid_particle_q75": relief,
        },
        "valid_particles": valid,
        "high_affinity_particles": high,
        "high_affinity_unique": high_unique,
        "quality_particles": quality,
        "quality_unique": quality_unique,
        "validation_selection": selected,
    }


def cohort_statistics(rows: list[Row]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": len(rows),
        "pb_full_pass": sum(flag(r, "pb_full_pass") for r in rows),
        "energy_converged": sum(r["energy_status"] == "converged" for r in rows),
        "sa_ge_3": sum(float(r["sa_score"]) >= 3 for r in rows),
        "sa_ge_4": sum(float(r["sa_score"]) >= 4 for r in rows),
    }
    for key in METRICS:
        values = np.array([float(r[key]) for r in rows])
        result[key] = {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return result


def sdf_blocks(path: Path) -> dict[str, bytes]:
    """Preserve even an unnamed molecule's blank first header line byte for byte."""
    blocks = {}
    buffer = bytearray()
    for line in path.read_bytes().splitlines(keepends=True):
        buffer.extend(line)
        if line.strip() != b"$$$$":
            continue
        raw = bytes(buffer)
        mol = next(Chem.ForwardSDMolSupplier(io.BytesIO(raw), removeHs=False))
        if mol is None or not mol.HasProp("node_id"):
            raise ArtifactError(f"Invalid original SDF record in {path.name}")
        node_id = mol.GetProp("node_id")
        if node_id in blocks:
            raise ArtifactError("Duplicate source SDF node ID")
        blocks[node_id] = raw
        buffer.clear()
    if buffer.strip():
        raise ArtifactError("Unterminated source SDF record")
    return blocks


def prepare_ck2_sa(workspace: Path, output: Path, count_per_stratum: int = 3) -> CaseDataset:
    scores = (
        workspace / "outputs/ck2_original_steer_analysis_20261013/tables/terminal_all_features.csv"
    )
    seeds = workspace / "outputs/ck2_top10_process_analysis_20261014/actual_generation_seed_map.csv"
    poses = workspace / "outputs/ck2_final_spatial_analysis_20261014/extracted"
    rows = read_csv(scores)
    groups = select_cohorts(rows, count_per_stratum)
    selected = groups["validation_selection"]
    seed_map = {int(r["batch"]): int(r["seed"]) for r in read_csv(seeds)}
    pose_files = [
        poses / f"valid_final_{batch:03d}.sdf"
        for batch in sorted({int(r["batch"]) for r in selected})
    ]
    blocks = {key: value for path in pose_files for key, value in sdf_blocks(path).items()}
    cases = []
    for index, row in enumerate(selected):
        if int(row["seed"]) != seed_map[int(row["batch"])]:
            raise ArtifactError("Recorded actual sampler seed disagrees with seed map")
        if row["node_id"] not in blocks:
            raise ArtifactError("Selected particle has no original pose")
        cases.append(
            SourceCase(
                case_id=row["node_id"],
                target="CK2",
                smiles=row["smiles"],
                selection_rank=index + 1,
                sdf_record_index=index,
                source_metrics={
                    **{k: float(row[k]) for k in METRICS},
                    **{
                        k: flag(row, k)
                        for k in ("quality_tail", "pb_full_pass", "strain_minimization_converged")
                    },
                },
                provenance={
                    "batch": int(row["batch"]),
                    "actual_sampler_seed": int(row["seed"]),
                    "root_slot": int(row["root_slot"]),
                    "cohort": row["validation_stratum"],
                    "stratum_rank": row["stratum_rank"],
                    "pose": "original raw decoded pose, source SDF bytes unchanged",
                },
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    sdf = output / "candidates.sdf"
    sdf.write_bytes(b"".join(blocks[c.case_id] for c in cases))
    dataset = CaseDataset(
        selection_rule="Original particle Q90 affinity and Q75 relief gates; independent canonical "
        "graphs; descending SA top N per quality and high-strain challenge stratum; "
        "challenge requires SA>=3, full PB pass, converged energy, no qualifying "
        "quality pose of the same graph. Select before agent calls.",
        source_files={
            p.relative_to(workspace).as_posix(): sha256(p) for p in (scores, seeds, *pose_files)
        },
        sdf_file=sdf.name,
        sdf_sha256=sha256(sdf),
        cases=cases,
    )
    write_json(output / "cases.json", dataset.model_dump(mode="json"))
    load_dataset(output / "cases.json")
    for name in (
        "high_affinity_particles",
        "high_affinity_unique",
        "quality_unique",
        "validation_selection",
    ):
        write_csv(output / f"{name}.csv", groups[name])
    write_json(
        output / "selection.json",
        {
            "created_at": utc_now(),
            "source_attempts": len(rows),
            "thresholds": groups["thresholds"],
            "selection_rule": dataset.selection_rule,
            "count_per_stratum": count_per_stratum,
            "statistics": {
                name: cohort_statistics(groups[name])
                for name in (
                    "valid_particles",
                    "high_affinity_particles",
                    "high_affinity_unique",
                    "quality_particles",
                    "quality_unique",
                )
            },
            "deduplication": "canonical isomeric SMILES, no tautomer or protonation merging",
            "representative": "highest affinity within each eligible cohort, then lowest relief, ID",
            "validation_case_ids": [c.case_id for c in cases],
            "route_status": "not_run",
            "source_files": dataset.source_files,
        },
    )
    for name in ("quality_unique", "high_affinity_unique"):
        targets = groups[name]
        (output / f"retro_{name}.smi").write_text(
            "".join(r["canonical_smiles"] + "\n" for r in targets), encoding="utf-8"
        )
        write_csv(
            output / f"retro_{name}_index.csv",
            [
                {
                    "target_index_zero_based": i,
                    "node_id": r["node_id"],
                    "smiles": r["canonical_smiles"],
                    "status": "not_run",
                }
                for i, r in enumerate(targets)
            ],
        )
    return dataset
