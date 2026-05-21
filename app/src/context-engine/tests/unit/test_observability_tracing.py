"""Phase B — event tracing: ingress links, resolve span, correlation flow."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from application.use_cases.context_graph_jobs import _ingress_links
from application.use_cases.resolve_context import resolve_context
from bootstrap import observability_runtime
from bootstrap.observability_context import get_correlation
from domain.intelligence_models import ContextResolutionRequest, IntelligenceBundle


class RecordingObs:
    """Minimal ObservabilityPort double that records what it saw."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, tuple]] = []
        self.counters: list[tuple[str, int]] = []
        self.histograms: list[tuple[str, float]] = []

    @contextmanager
    def span(self, name, *, kind="internal", attributes=None, links=None):
        self.spans.append((name, tuple(links or ())))

        class _S:
            def set_attribute(self, *a):
                pass

            def set_attributes(self, *a):
                pass

            def add_event(self, *a, **k):
                pass

            def record_exception(self, *a):
                pass

            def set_error(self, *a):
                pass

        yield _S()

    def current_traceparent(self):
        return None

    @contextmanager
    def baggage(self, **items):
        yield

    def counter(self, name, value=1, *, attributes=None):
        self.counters.append((name, value))

    def histogram(self, name, value, *, attributes=None):
        self.histograms.append((name, value))

    def gauge(self, name, value, *, attributes=None):
        pass


@pytest.fixture
def recording_obs(monkeypatch: pytest.MonkeyPatch) -> RecordingObs:
    obs = RecordingObs()
    monkeypatch.setattr(observability_runtime, "_OBSERVABILITY", obs)
    return obs


@pytest.mark.unit
def test_ingress_links_collects_persisted_traceparents() -> None:
    class Ref:
        def __init__(self, eid: str) -> None:
            self.event_id = eid

    class Repo:
        def list_events_for_batch(self, _bid: str):
            return [Ref("e1"), Ref("e2"), Ref("e3")]

    class Row:
        def __init__(self, cid: str | None) -> None:
            self.correlation_id = cid

    rows = {
        "e1": Row("00-aaaa-bbbb-01"),
        "e2": Row(None),  # never-traced event
        "e3": Row("00-cccc-dddd-01"),
    }

    class Ledger:
        def get_event_by_id(self, eid: str):
            return rows.get(eid)

    links = _ingress_links(Ledger(), Repo(), "batch-1")
    assert links == ["00-aaaa-bbbb-01", "00-cccc-dddd-01"]


@pytest.mark.unit
def test_ingress_links_is_best_effort_on_error() -> None:
    class BadRepo:
        def list_events_for_batch(self, _bid: str):
            raise RuntimeError("db down")

    assert _ingress_links(object(), BadRepo(), "b") == []


@pytest.mark.unit
async def test_resolve_context_opens_span_and_binds_pot(
    recording_obs: RecordingObs,
) -> None:
    seen_corr: dict = {}

    class FakeService:
        async def resolve(self, request):
            seen_corr.update(get_correlation())
            return IntelligenceBundle(request=request)

    req = ContextResolutionRequest(pot_id="pot-xyz", query="why")
    bundle = await resolve_context(FakeService(), req)

    assert bundle.request.pot_id == "pot-xyz"
    # pot bound for the duration of the awaited resolve
    assert seen_corr.get("pot_id") == "pot-xyz"
    # correlation cleaned up after the scope exits
    assert get_correlation() == {}
    # span + metrics emitted
    assert any(n == "context.resolve" for n, _ in recording_obs.spans)
    assert any(n == "ce.resolve.latency_ms" for n, _ in recording_obs.histograms)
    assert ("ce.resolve.total", 1) in recording_obs.counters


@pytest.mark.unit
async def test_resolve_context_records_error_metric(
    recording_obs: RecordingObs,
) -> None:
    class Boom:
        async def resolve(self, request):
            raise ValueError("nope")

    with pytest.raises(ValueError):
        await resolve_context(Boom(), ContextResolutionRequest(pot_id="p", query="q"))
    assert ("ce.resolve.total", 1) in recording_obs.counters
    assert get_correlation() == {}
