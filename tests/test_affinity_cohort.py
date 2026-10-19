import pytest

from agenticgen.affinity_cohort import affinity_ranking, prepare_ck2_affinity
from agenticgen.cohorts import read_csv, sdf_blocks
from agenticgen.errors import ArtifactError


def test_best_graph_representative_is_selected_before_quality_filtering():
    def row(name, smiles, score, **extra):
        return dict(
            node_id=name,
            smiles=smiles,
            pic50_on_rescore=score,
            arm="single",
            valid_connected=True,
            **extra,
        )

    rows = [
        row("low_quality_best", "OCC", 9.1, pb_full_pass=False),
        row("good_quality_duplicate", "CCO", 8.0, pb_full_pass=True),
        row("other", "CCN", 8.8, pb_full_pass=True),
        row("nan_score", "CCC", "nan", pb_full_pass=True),
        {**row("control", "CCCl", 10.0), "arm": "control"},
        {**row("invalid", "C", 12.0), "valid_connected": False},
    ]
    result = affinity_ranking(rows, 2)
    assert [r["node_id"] for r in result["selected"]] == ["low_quality_best", "other"]
    assert result["unique_graph_count"] == 2
    assert result["nonfinite_affinity_ids"] == ["nan_score"]
    with pytest.raises(ArtifactError, match="Insufficient"):
        affinity_ranking(rows, 3)


def test_original_global_top20_retains_best_pose_and_original_sdf(repo, tmp_path):
    workspace = repo.parents[1]
    source = (
        workspace / "outputs/ck2_original_steer_analysis_20261013/tables/terminal_all_features.csv"
    )
    if not source.exists():
        pytest.skip("Original CK2 data unavailable")
    rows = read_csv(source)
    selected = affinity_ranking(rows, 20)["selected"]
    assert len({r["canonical_smiles"] for r in selected}) == 20
    assert any(r["quality_tail"] == "False" for r in selected)
    dataset = prepare_ck2_affinity(workspace, tmp_path / "top20", 20)
    assert [c.case_id for c in dataset.cases] == [r["node_id"] for r in selected]
    exported = sdf_blocks(tmp_path / "top20/candidates.sdf")
    for case in dataset.cases:
        original = workspace / (
            "outputs/ck2_final_spatial_analysis_20261014/extracted/"
            f"valid_final_{case.provenance['batch']:03d}.sdf"
        )
        assert exported[case.case_id] == sdf_blocks(original)[case.case_id]
