import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.modules.intelligence.tools.impact_analysis.impact_analysis_config import (
    is_allowed_xml_path,
)
from app.modules.intelligence.tools.impact_analysis.impact_trace_tool import (
    ImpactTraceAnalysisTool,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "impact_analysis_golden_cases.json"


def _base_candidate(file_path: str) -> Dict[str, Any]:
    return {
        "node_id": "node-1",
        "name": "ChangedFunction",
        "file_path": file_path,
        "content": "void ChangedFunction() {}",
    }


def _configure_case(tool: ImpactTraceAnalysisTool, monkeypatch, scenario: str) -> None:
    monkeypatch.setattr(tool, "_expand_neighbours", lambda *args, **kwargs: [])

    if scenario == "ambiguous":
        monkeypatch.setattr(
            tool,
            "_resolve_function_candidates",
            lambda *args, **kwargs: [
                _base_candidate("src/module/service.cs"),
                _base_candidate("src/module/duplicate.cs"),
            ],
        )
    else:
        monkeypatch.setattr(
            tool,
            "_resolve_function_candidates",
            lambda *args, **kwargs: [_base_candidate("src/module/service.cs")],
        )

    if scenario == "partial":
        monkeypatch.setattr(
            tool,
            "_query_knowledge_graph",
            lambda *args, **kwargs: [
                {
                    "file_path": "tests/module/test_service.py",
                    "docstring": "test coverage",
                    "query": "What tests relate to ChangedFunction?",
                }
            ],
        )
    else:
        monkeypatch.setattr(tool, "_query_knowledge_graph", lambda *args, **kwargs: [])

    if scenario == "scope_blocked":
        monkeypatch.setattr(
            tool,
            "_collect_identifier_tokens",
            lambda *args, **kwargs: [
                {
                    "original": "AutomationId",
                    "variant": "AutomationId",
                    "normalized": "automationid",
                }
            ],
        )
    else:
        monkeypatch.setattr(tool, "_collect_identifier_tokens", lambda *args, **kwargs: [])

    def fake_search(_project_id: str, query: str) -> List[Dict[str, Any]]:
        if scenario == "scope_blocked" and query == "AutomationId":
            return [
                {
                    "file_path": "TestCode/FlaUITaskLayer/PrimaryDisplayUI/PrimaryDisplayControls.xml",
                    "content": "AutomationId=ScopedWidget",
                },
                {
                    "file_path": "_Release/UDD/Desktop/Source/OutOfScope/Blocked.xml",
                    "content": "AutomationId=BlockedWidget",
                },
            ]

        if scenario == "partial":
            # No direct test hit; rely on KG-only medium confidence recommendation.
            return []

        if "test" in query:
            return [
                {
                    "file_path": "tests/module/test_service.py",
                    "content": "ChangedFunction()",
                }
            ]

        return []

    monkeypatch.setattr(tool, "_search_codebase", fake_search)


@pytest.mark.parametrize(
    "case_data",
    json.loads(FIXTURE_PATH.read_text()),
    ids=lambda case: case["name"],
)
def test_impact_analysis_golden_cases(monkeypatch, case_data: Dict[str, Any]):
    tool = ImpactTraceAnalysisTool(sql_db=None, user_id="test-user")
    _configure_case(tool, monkeypatch, case_data["scenario"])

    result = tool.run(**case_data["input"])
    expected = case_data["expected"]

    # Golden expectations by scenario.
    assert len(result["recommended_tests"]) >= expected["min_recommended_tests"]
    assert bool(result["blocked_by_scope"]) == expected["expect_blocked_scope"]
    assert bool(result["ambiguities"]) == expected["expect_ambiguities"]

    # Every recommended test must have evidence.
    for recommended_test in result["recommended_tests"]:
        assert recommended_test["evidence_ids"]

    # strict_mode must exclude low-confidence recommendations.
    if case_data["input"].get("strict_mode", True):
        assert all(test["confidence"] != "low" for test in result["recommended_tests"])

    # No absolute paths in any output path-bearing fields.
    path_values = []
    path_values.extend(test["file_path"] for test in result["recommended_tests"])
    path_values.extend(evidence["file_path"] for evidence in result["evidence"])
    path_values.extend(item["file_path"] for item in result["blocked_by_scope"])

    for path in path_values:
        assert not path.startswith("/")
        assert not re.match(r"^[A-Za-z]:", path)

    # XML evidence must stay inside allowlisted roots.
    for evidence in result["evidence"]:
        if evidence["type"] == "xml_mapping":
            assert is_allowed_xml_path(evidence["file_path"])
