from agenticgen.reporting import summarize_results


def test_constraint_failure_does_not_count_as_an_executed_route_search():
    results = [
        {
            "case_id": "a",
            "status": "completed",
            "candidates": [{"status": "rejected", "constrained_route_success": False}],
        },
        {
            "case_id": "b",
            "status": "completed",
            "candidates": [
                {
                    "status": "accepted",
                    "constrained_route_success": None,
                    "retrosynthesis": {"status": "not_run"},
                }
            ],
        },
    ]
    summary = summarize_results(results, "unit_test")
    assert summary["primary_route_evaluated_count"] == 0
    assert summary["primary_constraint_failure_count"] == 1
    assert summary["primary_outcome_available_count"] == 1


def test_unverified_search_route_does_not_become_qualified_success():
    results = [
        {
            "case_id": "a",
            "status": "completed",
            "candidates": [
                {
                    "status": "accepted",
                    "constrained_route_success": None,
                    "retrosynthesis": {"status": "route_found", "route_verified": False},
                }
            ],
        }
    ]
    summary = summarize_results(results, "unit_test")
    assert summary["primary_route_evaluated_count"] == 1
    assert summary["primary_route_success_count"] == 0
    assert summary["primary_outcome_available_count"] == 0
