"""Post-Graphiti ontology classifier pass on a mock Neo4j driver."""

from __future__ import annotations

import pytest

from adapters.outbound.graphiti.ontology_classifier_pass import (
    run_ontology_classifier_pass,
)

pytestmark = pytest.mark.unit


class _FakeDriver:
    """Simulates a Neo4j driver for the classifier pass."""

    def __init__(
        self,
        nodes: list[dict],
        edges: list[dict],
        *,
        provider=None,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self.set_calls: list[tuple[str, list[str]]] = []
        try:
            from graphiti_core.driver.driver import GraphProvider

            self.provider = provider or GraphProvider.NEO4J
        except Exception:  # pragma: no cover
            self.provider = provider

    async def execute_query(self, query: str, **params):
        if "labels(n) AS labels" in query and "properties(n)" in query:
            rows = [
                {
                    "uuid": n["uuid"],
                    "labels": n.get("labels", []),
                    "props": n.get("props", {}),
                }
                for n in self._nodes
            ]
            return rows, None, None
        if "RELATES_TO" in query and "RETURN a.uuid AS src" in query:
            rows = [
                {
                    "src": e["src"],
                    "tgt": e["tgt"],
                    "name": e["name"].upper().strip(),
                }
                for e in self._edges
            ]
            return rows, None, None
        if "SET n:" in query:
            label = query.split("SET n:", 1)[1].split()[0].strip()
            uuids = list(params.get("uuids", []))
            self.set_calls.append((label, uuids))
            hits = [n for n in self._nodes if n["uuid"] in uuids]
            return [{"cnt": len(hits)}], None, None
        return [], None, None


def _graphiti_available() -> bool:
    try:
        from graphiti_core.driver.driver import GraphProvider  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _graphiti_available(), reason="graphiti_core not installed")
async def test_pass_adds_decision_from_text_and_edge(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    driver = _FakeDriver(
        nodes=[
            {
                "uuid": "n1",
                "labels": ["Entity", "Feature"],
                "props": {
                    "name": "Background workers",
                    "summary": "We decided to adopt Hatchet over Celery.",
                },
            },
            {
                "uuid": "n2",
                "labels": ["Entity"],
                "props": {"name": "ledger"},
            },
        ],
        edges=[
            {"src": "n1", "tgt": "n2", "name": "AFFECTS"},
        ],
    )

    result = await run_ontology_classifier_pass(driver, "pot-1")

    assert result["ok"] is True
    assert result["entities_classified"] >= 1
    labels_set = {label for label, _ in driver.set_calls}
    assert "Decision" in labels_set


@pytest.mark.skipif(not _graphiti_available(), reason="graphiti_core not installed")
async def test_pass_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")
    driver = _FakeDriver(nodes=[], edges=[])
    result = await run_ontology_classifier_pass(driver, "pot-1")
    assert result.get("skipped") == "disabled"


@pytest.mark.skipif(not _graphiti_available(), reason="graphiti_core not installed")
async def test_pass_forces_through_disabled_flag(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")
    driver = _FakeDriver(
        nodes=[
            {
                "uuid": "n1",
                "labels": ["Entity"],
                "props": {"canonical_type": "Runbook", "title": "r1"},
            }
        ],
        edges=[],
    )
    result = await run_ontology_classifier_pass(driver, "pot-1", force=True)
    assert result["ok"] is True
    assert any(label == "Runbook" for label, _ in driver.set_calls)


@pytest.mark.skipif(not _graphiti_available(), reason="graphiti_core not installed")
async def test_pass_infers_fix_from_resolved_edge(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    driver = _FakeDriver(
        nodes=[
            {"uuid": "fix", "labels": ["Entity"], "props": {"name": "retry patch"}},
            {
                "uuid": "inc",
                "labels": ["Entity"],
                "props": {"name": "webhook outage"},
            },
        ],
        edges=[{"src": "fix", "tgt": "inc", "name": "RESOLVED"}],
    )
    await run_ontology_classifier_pass(driver, "pot-1")
    per_label = {label: uuids for label, uuids in driver.set_calls}
    # Edge rule pins the source of RESOLVED to Fix (target is ambiguous).
    assert "fix" in per_label.get("Fix", [])
    assert "inc" not in per_label.get("Fix", [])
