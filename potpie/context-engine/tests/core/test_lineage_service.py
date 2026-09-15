"""SQLite lineage store + LineageService reverse query."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.application.services.lineage_service import (
    LineageService,
    mint_prompt_key,
)

pytestmark = pytest.mark.unit


def test_why_returns_prompt_and_spec_hashes(tmp_path: Path) -> None:
    target = tmp_path / "foo.py"
    target.write_text("a\n" * 50, encoding="utf-8")
    service = LineageService.for_pot("demo", home=tmp_path, fail_open=True)
    prompt = "Please add retry around the HTTP client."
    spec = "The client must retry transient 5xx with backoff."
    captured = service.capture(
        path=str(target),
        line_start=12,
        line_end=40,
        prompt=prompt,
        spec=spec,
        harness="claude",
        session_id="sess-1",
        repo="acme/api",
    )
    assert captured["ok"]
    assert captured["prompt_hash"]
    assert captured["spec_hash"]
    assert captured["session_key"].startswith("session:claude:")

    result = service.why(path=str(target), line_start=12, line_end=40)
    assert result["matches"]
    hit = result["matches"][0]
    assert hit["prompt_hash"] == captured["prompt_hash"]
    assert hit["spec_hash"] == captured["spec_hash"]
    assert hit["session_key"] == captured["session_key"]
    assert hit["prompt"] == prompt
    assert hit["spec"] == spec
    assert "l12-l40-" in hit["code_asset_key"]


def test_why_overlaps_partial_line_range(tmp_path: Path) -> None:
    service = LineageService.for_pot("demo", home=tmp_path)
    service.remember_prompt(prompt="edit the handler", session_id="s", harness="claude")
    service.capture(
        path="src/app.py",
        line_start=1,
        line_end=80,
        session_id="s",
        harness="claude",
        span_text="whole file",
    )
    result = service.why(path="src/app.py", line_start=12, line_end=40)
    assert result["matches"]
    assert result["matches"][0]["prompt_hash"] == mint_prompt_key(
        "edit the handler"
    ).rsplit(":", 1)[-1]


def test_graph_record_failure_is_fail_open(tmp_path: Path) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("graph down")

    service = LineageService.for_pot(
        "demo", home=tmp_path, record_graph=boom, fail_open=True
    )
    result = service.capture(
        path="foo.py",
        line_start=1,
        line_end=2,
        prompt="do the thing",
        span_text="x",
    )
    assert result["ok"]
    assert result["graph_error"]
    why = service.why(path="foo.py", line_start=1, line_end=2)
    assert why["matches"][0]["prompt_hash"] == result["prompt_hash"]


def test_graph_service_records_generation_link(tmp_path: Path) -> None:
    from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
        InMemoryGraphBackend,
    )
    from potpie_context_engine.application.services.graph_service import (
        DefaultGraphService,
    )
    from potpie_context_engine.core.ports.agent_context import RecordRequest
    from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter

    svc = DefaultGraphService(backend=InMemoryGraphBackend())
    recorded: list[str] = []

    def record_graph(record_type, summary, details, scope):
        receipt = svc.record(
            RecordRequest(
                pot_id="demo",
                record_type=record_type,
                summary=summary,
                details=details,
                scope=scope,
            )
        )
        recorded.append(record_type)
        assert receipt.accepted, receipt

    service = LineageService.for_pot(
        "demo", home=tmp_path, record_graph=record_graph, fail_open=False
    )
    result = service.capture(
        path="src/foo.py",
        line_start=12,
        line_end=40,
        prompt="add retries",
        spec="retry 5xx with backoff",
        harness="claude",
        session_id="sess-1",
        repo="acme/api",
        span_text="def retry():\n    pass",
    )
    assert "prompt_turn" in recorded
    assert "generation_link" in recorded
    rows = svc.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="demo", predicate_in=("GENERATED_FROM", "IMPLEMENTS"))
    )
    predicates = {row.predicate for row in rows}
    assert "GENERATED_FROM" in predicates
    assert "IMPLEMENTS" in predicates
    assert result["code_asset_key"].startswith("code:")

    why_service = LineageService.for_pot(
        "demo",
        home=tmp_path,
        claim_query=svc.backend.claim_query,
        fail_open=False,
    )
    why = why_service.why(path="src/foo.py", line_start=12, line_end=40)
    assert why["matches"]
    claims = why["matches"][0].get("claims") or []
    claim_preds = {c["predicate"] for c in claims}
    assert "GENERATED_FROM" in claim_preds
    assert "IMPLEMENTS" in claim_preds
    assert any(
        c["subject_key"] == result["code_asset_key"]
        or c["object_key"] == result["code_asset_key"]
        for c in claims
    )

