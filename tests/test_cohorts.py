import io

import numpy as np
import pytest
from rdkit import Chem

from agenticgen.cohorts import prepare_ck2_sa, read_csv, sdf_blocks, select_cohorts
from agenticgen.dataset import load_dataset
from agenticgen.errors import ArtifactError


def test_selection_matches_original_gates_and_separates_challenge_graphs(repo, tmp_path):
    workspace = repo.parents[1]
    source = (
        workspace / "outputs/ck2_original_steer_analysis_20261013/tables/terminal_all_features.csv"
    )
    if not source.exists():
        pytest.skip("Original CK2 data not available")
    rows = read_csv(source)
    groups = select_cohorts(rows)
    assert len(groups["high_affinity_unique"]) == 45
    assert len(groups["quality_unique"]) == 24
    assert [r["node_id"] for r in groups["validation_selection"]] == [
        "single_b017_final_021",
        "single_b004_final_020",
        "single_b004_final_008",
        "single_b011_final_017",
        "single_b000_final_034",
        "single_b011_final_016",
    ]
    assert set(r["canonical_smiles"] for r in groups["validation_selection"][3:]).isdisjoint(
        r["canonical_smiles"] for r in groups["quality_unique"]
    )
    output = tmp_path / "cohort"
    dataset = prepare_ck2_sa(workspace, output)
    assert load_dataset(output / "cases.json") == dataset
    exported = sdf_blocks(output / "candidates.sdf")
    for case in dataset.cases:
        source_sdf = workspace / (
            "outputs/ck2_final_spatial_analysis_20261014/extracted/"
            f"valid_final_{case.provenance['batch']:03d}.sdf"
        )
        original = sdf_blocks(source_sdf)[case.case_id]
        assert exported[case.case_id] == original
        mol = next(Chem.ForwardSDMolSupplier(io.BytesIO(original)))
        assert np.isfinite(mol.GetConformer().GetPositions()).all()
    with pytest.raises(FileExistsError):
        prepare_ck2_sa(workspace, output)
    # Detect a changed historic label rather than silently changing the population.
    next(r for r in rows if r["quality_tail"] == "True")["quality_tail"] = "False"
    with pytest.raises(ArtifactError, match="Quality gate"):
        select_cohorts(rows)
