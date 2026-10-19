"""Freeze the previously selected CK2 examples without re-ranking or altering their poses."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from rdkit import Chem

from agenticgen.chemistry import canonical_smiles
from agenticgen.contracts import CaseDataset, SourceCase
from agenticgen.errors import ArtifactError
from agenticgen.io import read_json, sha256, write_json

METRICS = (
    "pic50_on_rescore",
    "pic50_off_rescore",
    "sa_score",
    "mmff_relief_per_heavy",
    "mmff_relief_kcal_mol",
    "strain_MMFF94s_kcal_mol",
)


def load_dataset(path: Path) -> CaseDataset:
    dataset = CaseDataset.model_validate(read_json(path))
    if not dataset.cases or len({c.case_id for c in dataset.cases}) != len(dataset.cases):
        raise ArtifactError("Dataset requires nonempty unique case IDs")
    sdf = (path.parent / dataset.sdf_file).resolve()
    if sha256(sdf) != dataset.sdf_sha256:
        raise ArtifactError("Example SDF checksum mismatch")
    records = list(Chem.ForwardSDMolSupplier(io.BytesIO(sdf.read_bytes()), removeHs=False))
    if len(records) != len(dataset.cases):
        raise ArtifactError("SDF record count differs from the case manifest")
    if sorted(c.sdf_record_index for c in dataset.cases) != list(range(len(records))):
        raise ArtifactError("Each SDF record must belong to exactly one case")
    for case in dataset.cases:
        mol = records[case.sdf_record_index]
        if mol is None or not mol.HasProp("node_id") or mol.GetProp("node_id") != case.case_id:
            raise ArtifactError("Case and SDF record identities disagree")
        if canonical_smiles(mol) != canonical_smiles(case.smiles):
            raise ArtifactError("Case and SDF molecular graphs disagree")
        if not mol.GetNumConformers():
            raise ArtifactError("A source pose is required for the CK2 example")
    return dataset


def prepare_ck2(workspace: Path, output: Path, count: int = 3) -> CaseDataset:
    if not 1 <= count <= 10:
        raise ValueError("Select 1 to 10 members of the frozen TOP10 list")
    original = workspace / "outputs/ck2_original_steer_analysis_20261013"
    top10 = workspace / "outputs/ck2_top10_process_analysis_20261014"
    selected = top10 / "selected_candidates.csv"
    scores = original / "tables/terminal_all_features.csv"
    sdf_source = top10 / "top10_unique_quality_graphs.sdf"
    with selected.open(encoding="utf-8", newline="") as stream:
        rows = sorted(
            (row for row in csv.DictReader(stream) if row["cohort"] == "top10_unique_graphs"),
            key=lambda row: int(row["selection_rank"]),
        )[:count]
    if [int(row["selection_rank"]) for row in rows] != list(range(1, count + 1)):
        raise ArtifactError("Missing frozen TOP10 selection ranks")
    with scores.open(encoding="utf-8", newline="") as stream:
        metrics = {row["node_id"]: row for row in csv.DictReader(stream)}
    # Preserve each original mol block and its coordinate precision byte for byte.
    blocks: dict[str, bytes] = {}
    for raw in sdf_source.read_bytes().split(b"$$$$"):
        if not raw.strip():
            continue
        raw = raw.lstrip(b"\r\n") + b"$$$$\n"
        mol = next(Chem.ForwardSDMolSupplier(io.BytesIO(raw), removeHs=False))
        if mol is None:
            raise ArtifactError("Invalid source SDF record")
        blocks[mol.GetProp("node_id")] = raw
    cases = []
    for index, row in enumerate(rows):
        record = metrics[row["node_id"]]
        if record["quality_tail"] != "True" or record["pb_full_pass"] != "True":
            raise ArtifactError("Frozen candidate no longer satisfies its recorded quality group")
        if canonical_smiles(record["smiles"]) != canonical_smiles(row["smiles"]):
            raise ArtifactError("Frozen candidate graph changed")
        cases.append(
            SourceCase(
                case_id=row["node_id"],
                target="CK2",
                smiles=row["smiles"],
                selection_rank=int(row["selection_rank"]),
                sdf_record_index=index,
                source_metrics={
                    **{name: float(record[name]) for name in METRICS},
                    "quality_tail": True,
                    "pb_full_pass": True,
                    "strain_minimization_converged": record["strain_minimization_converged"]
                    == "True",
                },
                provenance={
                    "batch": int(row["batch"]),
                    "actual_sampler_seed": int(row["seed"]),
                    "root_slot": int(row["root_slot"]),
                    "cohort": row["cohort"],
                    "pose": "original raw decoded pose, coordinates unchanged",
                    "selection": "frozen unique-graph rank, not a new affinity computation",
                },
            )
        )
    if (output / "cases.json").exists() or (output / "candidates.sdf").exists():
        raise FileExistsError("Prepared examples already exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    sdf_out = output / "candidates.sdf"
    sdf_out.write_bytes(b"".join(blocks[c.case_id] for c in cases))
    dataset = CaseDataset(
        selection_rule=(
            "First three frozen TOP10 unique quality graphs"
            if count == 3
            else f"First {count} frozen TOP10 unique quality graphs"
        )
        + "; original quality gate: upper affinity decile, full PoseBusters pass, "
        "converged MMFF relaxation, relief/heavy-atom <= original valid-cohort Q75.",
        source_files={
            p.relative_to(workspace).as_posix(): sha256(p) for p in (selected, scores, sdf_source)
        },
        sdf_file=sdf_out.name,
        sdf_sha256=sha256(sdf_out),
        cases=cases,
    )
    write_json(output / "cases.json", dataset.model_dump(mode="json"))
    return load_dataset(output / "cases.json")
