from benchmarks.evaluator import evaluate_response, grade, latency_stats


def test_evaluator_scores_keywords_paths_and_envelope_fields() -> None:
    response = {
        "answer": {"summary": "context_resolve uses source refs and ResolverBudget"},
        "coverage": {"status": "complete"},
        "quality": {"status": "watch"},
        "source_refs": [{"ref": "github:pr:potpie/api:42"}],
        "fallbacks": [],
    }
    scenario = {
        "expected": {
            "must_contain": ["context_resolve", "ResolverBudget"],
            "required_paths": ["answer.summary", "coverage"],
            "coverage": {"status": "complete"},
            "quality": {"status": "watch"},
            "min_source_refs": 1,
            "max_fallbacks": 0,
        }
    }

    score, max_score, checks = evaluate_response(response, scenario)

    assert score == max_score
    assert all(check["passed"] for check in checks)
    assert grade(score, max_score) == "excellent"


def test_latency_stats_percentiles() -> None:
    stats = latency_stats([1.0, 2.0, 3.0, 4.0])

    assert stats.min_ms == 1.0
    assert stats.p50_ms == 2.5
    assert stats.p95_ms == 3.85
    assert stats.max_ms == 4.0
