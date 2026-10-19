from agenticgen.base_config import SelectionConfig
from agenticgen.inpainting_selection import select_poses


def test_route_reachability_does_not_bypass_syncraft_binding_loss_budget():
    case = {
        "source_score_only": {
            "status": "ok",
            "protocol_id": "fixture",
            "graph_smiles": "CCO",
            "score_kcal_mol": -11,
        },
        "proposals": [
            {
                "proposal_index": 2,
                "evaluations": [
                    {
                        "status": "measured",
                        "pose_id": "fragment_0000",
                        "mapped_smiles": "CO",
                        "source_similarity": 0.7,
                        "properties": {"status": "ok", "value": 2},
                        "binding_score_only": {"status": "ok", "score_kcal_mol": -8},
                        "retrosynthesis": {"status": "route_found", "route_verified": True},
                    }
                ],
            }
        ],
    }
    settings = SelectionConfig(policy="skill_sa_binding", similarity_threshold=0.2)
    result = select_poses(case, settings)
    assert result["selected_pose_id"] is result["route_validated_pose_id"] is None
    assert result["primary"]["excluded"][0]["reason"] == "vina_loss_budget"
    case["proposals"][0]["evaluations"][0]["binding_score_only"]["score_kcal_mol"] = -10.5
    result = select_poses(case, settings)
    assert (
        result["selected_pose_id"]
        == result["route_validated_pose_id"]
        == "proposal_02/fragment_0000"
    )
