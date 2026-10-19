"""Freeze affinity-ranked CK2 graphs before any optimization outcomes are seen."""

from __future__ import annotations

import math
from pathlib import Path

from agenticgen.chemistry import canonical_smiles
from agenticgen.cohorts import flag, read_csv, sdf_blocks, write_csv
from agenticgen.contracts import CaseDataset, SourceCase
from agenticgen.dataset import METRICS, load_dataset
from agenticgen.errors import ArtifactError
from agenticgen.io import sha256, utc_now, write_json


def affinity_ranking(rows: list[dict], count: int = 20) -> dict:
    """Deduplicate all valid single-Steer graphs by their best finite pIC50.

    PB/strain/SA are reported, never selection gates. Exact affinity ties use ID;
    stereochemistry is retained and tautomers/protonation states are not merged.
    """
    if count < 1:
        raise ValueError("Candidate count must be positive")
    if len({r["node_id"] for r in rows}) != len(rows):
        raise ArtifactError("Duplicate particle IDs in source table")
    eligible, excluded = [], []
    for row in rows:
        if row["arm"] != "single" or not flag(row, "valid_connected"):
            continue
        score = float(row["pic50_on_rescore"])
        if not math.isfinite(score):
            excluded.append(row["node_id"])
            continue
        eligible.append({**row, "canonical_smiles": canonical_smiles(row["smiles"])})
    by_graph = {}
    for row in sorted(eligible, key=lambda r: (-float(r["pic50_on_rescore"]), r["node_id"])):
        by_graph.setdefault(row["canonical_smiles"], row)
    if len(by_graph) < count:
        raise ArtifactError("Insufficient distinct valid graphs; selection is not relaxed")
    return {
        "source_particle_count": len(rows),
        "eligible_particle_count": len(eligible),
        "unique_graph_count": len(by_graph),
        "nonfinite_affinity_ids": excluded,
        "selected": [
            {**row, "affinity_rank": i + 1} for i, row in enumerate(list(by_graph.values())[:count])
        ],
    }


def prepare_ck2_affinity(workspace: Path, output: Path, count: int = 20) -> CaseDataset:
    scores = (
        workspace / "outputs/ck2_original_steer_analysis_20261013/tables/terminal_all_features.csv"
    )
    seeds = workspace / "outputs/ck2_top10_process_analysis_20261014/actual_generation_seed_map.csv"
    poses = workspace / "outputs/ck2_final_spatial_analysis_20261014/extracted"
    ranking = affinity_ranking(read_csv(scores), count)
    selected = ranking["selected"]
    seed_map = {int(r["batch"]): int(r["seed"]) for r in read_csv(seeds)}
    pose_files = [
        poses / f"valid_final_{batch:03d}.sdf"
        for batch in sorted({int(r["batch"]) for r in selected})
    ]
    blocks = {key: value for path in pose_files for key, value in sdf_blocks(path).items()}
    cases = []
    for index, row in enumerate(selected):
        if int(row["seed"]) != seed_map[int(row["batch"])]:
            raise ArtifactError("Sampler seed differs from original seed map")
        if row["node_id"] not in blocks:
            raise ArtifactError("Selected source particle has no original SDF")
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
                    "cohort": "global_affinity_unique_graphs",
                    "pose": "original source SDF bytes unchanged",
                    "representative": "highest pIC50 of the graph across all valid single-Steer particles",
                },
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    sdf = output / "candidates.sdf"
    sdf.write_bytes(b"".join(blocks[c.case_id] for c in cases))
    dataset = CaseDataset(
        selection_rule=f"Top {count} canonical isomeric graphs among all valid single-Steer "
        "particles; retain each graph's highest pic50_on_rescore, exact ties by particle ID. "
        "No PB, strain or SA gate. No tautomer/protonation merging. Frozen before API calls.",
        source_files={
            p.relative_to(workspace).as_posix(): sha256(p) for p in (scores, seeds, *pose_files)
        },
        sdf_file=sdf.name,
        sdf_sha256=sha256(sdf),
        cases=cases,
    )
    write_json(output / "cases.json", dataset.model_dump(mode="json"))
    write_csv(output / "selected_candidates.csv", selected)
    write_json(
        output / "selection.json",
        {
            "created_at": utc_now(),
            **{k: v for k, v in ranking.items() if k != "selected"},
            "count": count,
            "selection_rule": dataset.selection_rule,
            "selected_ids": [c.case_id for c in cases],
            "source_files": dataset.source_files,
        },
    )
    return load_dataset(output / "cases.json")
