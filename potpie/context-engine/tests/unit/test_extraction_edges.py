"""Graphiti extraction normalization (edge-type collapse)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters.outbound.graphiti.edge_extraction_normalize import (
    normalize_graphiti_extracted_edges,
)
from domain.extraction_edges import (
    infer_lifecycle_status,
    is_legitimate_pr_code_modified,
)


def test_infer_lifecycle_planned() -> None:
    assert (
        infer_lifecycle_status("OpenTelemetry spans will be added to the ingest path.")
        == "planned"
    )


def test_infer_lifecycle_decommissioned() -> None:
    assert (
        infer_lifecycle_status("The MongoDB cluster was decommissioned last quarter.")
        == "decommissioned"
    )


def test_infer_lifecycle_completed_migration() -> None:
    assert (
        infer_lifecycle_status("The ledger service was migrated from MongoDB to Postgres.")
        == "completed"
    )


def test_legitimate_modified_pr_to_file() -> None:
    assert is_legitimate_pr_code_modified(("PullRequest", "Entity"), ("FILE", "Entity"))


def test_vague_modified_remapped_to_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_STRICT_EXTRACTION", "0")
    edge = SimpleNamespace(
        name="MODIFIED",
        fact="Ledger service was migrated from MongoDB to Postgres.",
        attributes={},
        source_node_uuid="s1",
        target_node_uuid="t1",
    )
    nodes = [
        SimpleNamespace(uuid="s1", labels=["Service", "Entity"]),
        SimpleNamespace(uuid="t1", labels=["DataStore", "Entity"]),
    ]
    out = normalize_graphiti_extracted_edges([edge], nodes)
    assert out[0].name == "MIGRATED_TO"
    assert out[0].attributes.get("lifecycle_status") == "completed"


def test_legitimate_modified_keeps_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_STRICT_EXTRACTION", "0")
    edge = SimpleNamespace(
        name="MODIFIED",
        fact="Touches src/foo.py",
        attributes={},
        source_node_uuid="pr",
        target_node_uuid="f1",
    )
    nodes = [
        SimpleNamespace(uuid="pr", labels=["PullRequest", "Entity"]),
        SimpleNamespace(uuid="f1", labels=["FILE", "Entity"]),
    ]
    out = normalize_graphiti_extracted_edges([edge], nodes)
    assert out[0].name == "MODIFIED"


def test_strict_extraction_raises_on_high_vague_modified_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_STRICT_EXTRACTION", "1")
    edge = SimpleNamespace(
        name="MODIFIED",
        fact="Ledger service was migrated from MongoDB to Postgres.",
        attributes={},
        source_node_uuid="s1",
        target_node_uuid="t1",
    )
    nodes = [
        SimpleNamespace(uuid="s1", labels=["Service", "Entity"]),
        SimpleNamespace(uuid="t1", labels=["DataStore", "Entity"]),
    ]
    with pytest.raises(RuntimeError, match="High vague MODIFIED rate"):
        normalize_graphiti_extracted_edges([edge], nodes)
