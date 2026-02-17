from typing import Any, Dict, List

from app.modules.intelligence.tools.impact_analysis.impact_trace_tool import (
    ImpactTraceAnalysisTool,
)


def _stub_candidate() -> List[Dict[str, Any]]:
    return [
        {
            "node_id": "node-1",
            "name": "ChangedFunction",
            "file_path": "src/module/service.cs",
            "content": "void ChangedFunction() {}",
        }
    ]


def test_scope_enforcement_adds_blocked_by_scope_and_filters_evidence(monkeypatch):
    tool = ImpactTraceAnalysisTool(sql_db=None, user_id="test-user")

    monkeypatch.setattr(tool, "_resolve_function_candidates", lambda *args, **kwargs: _stub_candidate())
    monkeypatch.setattr(tool, "_query_knowledge_graph", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_expand_neighbours", lambda *args, **kwargs: [])
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

    def fake_search(_project_id: str, query: str):
        if query == "AutomationId":
            return [
                {
                    "file_path": "TestCode/FlaUITaskLayer/PrimaryDisplayUI/PrimaryDisplayControls.xml",
                    "content": "AutomationId=CustomerWidget",
                },
                {
                    "file_path": "_Release/UDD/Desktop/Source/PrimeDisp/View/Blocked.xml",
                    "content": "AutomationId=BlockedWidget",
                },
            ]
        if "test" in query:
            return [
                {
                    "file_path": "tests/module/test_service.py",
                    "content": "def test_changed_function(): ChangedFunction()",
                }
            ]
        return []

    monkeypatch.setattr(tool, "_search_codebase", fake_search)

    result = tool.run(
        project_id="project-id",
        changed_file="src/module/service.cs",
        function_name="ChangedFunction",
        strict_mode=True,
    )

    blocked_paths = {item["file_path"] for item in result["blocked_by_scope"]}
    evidence_paths = {item["file_path"] for item in result["evidence"]}

    assert "_Release/UDD/Desktop/Source/PrimeDisp/View/Blocked.xml" in blocked_paths
    assert "_Release/UDD/Desktop/Source/PrimeDisp/View/Blocked.xml" not in evidence_paths
    assert any(item["type"] == "xml_mapping" for item in result["evidence"])


def test_strict_mode_filters_low_confidence_heuristic_tests(monkeypatch):
    tool = ImpactTraceAnalysisTool(sql_db=None, user_id="test-user")

    monkeypatch.setattr(tool, "_resolve_function_candidates", lambda *args, **kwargs: _stub_candidate())
    monkeypatch.setattr(tool, "_query_knowledge_graph", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_expand_neighbours", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_collect_identifier_tokens", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_search_codebase", lambda *args, **kwargs: [])

    strict_result = tool.run(
        project_id="project-id",
        changed_file="src/module/service.cs",
        function_name="ChangedFunction",
        strict_mode=True,
    )
    non_strict_result = tool.run(
        project_id="project-id",
        changed_file="src/module/service.cs",
        function_name="ChangedFunction",
        strict_mode=False,
    )

    assert strict_result["recommended_tests"] == []
    assert non_strict_result["recommended_tests"]
    assert all(
        test["confidence"] == "low" for test in non_strict_result["recommended_tests"]
    )


def test_confidence_assignment_and_evidence_chain_shape(monkeypatch):
    tool = ImpactTraceAnalysisTool(sql_db=None, user_id="test-user")

    monkeypatch.setattr(tool, "_resolve_function_candidates", lambda *args, **kwargs: _stub_candidate())
    monkeypatch.setattr(tool, "_query_knowledge_graph", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_expand_neighbours", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_collect_identifier_tokens", lambda *args, **kwargs: [])

    def fake_search(_project_id: str, query: str):
        if "test" in query:
            return [
                {
                    "file_path": "tests/module/test_service.py",
                    "content": "ChangedFunction()",
                }
            ]
        return []

    monkeypatch.setattr(tool, "_search_codebase", fake_search)

    result = tool.run(
        project_id="project-id",
        changed_file="src/module/service.cs",
        function_name="ChangedFunction",
        strict_mode=True,
    )

    assert result["recommended_tests"]
    assert result["recommended_tests"][0]["confidence"] == "high"

    evidence_ids = {evidence["id"] for evidence in result["evidence"]}
    for recommended_test in result["recommended_tests"]:
        assert recommended_test["evidence_ids"]
        assert set(recommended_test["evidence_ids"]).issubset(evidence_ids)


def test_file_only_mode_returns_tests_without_function_name(monkeypatch):
    """When only changed_file is provided, tool should find tests via file-scoped search."""
    tool = ImpactTraceAnalysisTool(sql_db=None, user_id="test-user")

    def fake_search(_project_id: str, query: str):
        if "test" in query:
            return [
                {
                    "file_path": "tests/module/test_service.py",
                    "name": "test_changed_function",
                    "content": "def test_changed_function(): ...",
                }
            ]
        return []

    monkeypatch.setattr(tool, "_search_codebase", fake_search)

    result = tool.run(
        project_id="project-id",
        changed_file="src/module/service.cs",
        function_name=None,
        strict_mode=True,
    )

    assert result["recommended_tests"]
    assert result["recommended_tests"][0]["file_path"] == "tests/module/test_service.py"
    assert "test_ids" in result["recommended_tests"][0]
    test_ids = result["recommended_tests"][0]["test_ids"]
    assert test_ids
    assert any("test_changed_function" in tid for tid in test_ids)


def test_test_ids_populated_from_search_hit_content(monkeypatch):
    """Verify test_ids are extracted from def test_* in search hit content."""
    tool = ImpactTraceAnalysisTool(sql_db=None, user_id="test-user")

    monkeypatch.setattr(tool, "_resolve_function_candidates", lambda *args, **kwargs: _stub_candidate())
    monkeypatch.setattr(tool, "_query_knowledge_graph", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_expand_neighbours", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_collect_identifier_tokens", lambda *args, **kwargs: [])

    def fake_search(_project_id: str, query: str):
        if "test" in query:
            return [
                {
                    "file_path": "tests/test_foo.py",
                    "content": "def test_bar(): pass\ndef test_baz(): pass",
                }
            ]
        return []

    monkeypatch.setattr(tool, "_search_codebase", fake_search)

    result = tool.run(
        project_id="project-id",
        changed_file="src/foo.py",
        function_name="foo",
        strict_mode=True,
    )

    assert result["recommended_tests"]
    rec = result["recommended_tests"][0]
    assert rec["file_path"] == "tests/test_foo.py"
    assert "test_ids" in rec
    assert "tests/test_foo.py::test_bar" in rec["test_ids"]
    assert "tests/test_foo.py::test_baz" in rec["test_ids"]


def test_robot_and_specflow_test_ids_are_extracted():
    robot_hit = {
        "file_path": "TestCode/RobotTestcaseLayer/Regression/PrimaryDisplay.robot",
        "content": "StepLoginApplication_FlaUI\nStepOpenPrimaryDisplay_FlaUI",
    }
    feature_hit = {
        "file_path": "_Release/Platform/VMM/VMM/VMM.Functional.Tests/PrimaryDisplay.feature",
        "content": "Scenario: User logs in\nGiven(\"user logs in\")",
    }

    robot_names = ImpactTraceAnalysisTool._extract_test_names_from_search_hit(
        robot_hit["file_path"], robot_hit
    )
    feature_names = ImpactTraceAnalysisTool._extract_test_names_from_search_hit(
        feature_hit["file_path"], feature_hit
    )

    robot_ids = ImpactTraceAnalysisTool._build_test_ids(
        robot_hit["file_path"], set(robot_names)
    )
    feature_ids = ImpactTraceAnalysisTool._build_test_ids(
        feature_hit["file_path"], set(feature_names)
    )

    assert "StepLoginApplication_FlaUI" in robot_names
    assert any(test_id.endswith("keyword:StepLoginApplication_FlaUI") for test_id in robot_ids)
    assert "User logs in" in feature_names
    assert any(test_id.endswith("scenario:User logs in") for test_id in feature_ids)
