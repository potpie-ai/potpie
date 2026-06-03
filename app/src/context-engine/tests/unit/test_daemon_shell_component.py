"""ContextGraphComponent + ops: delegate to HostShell.agent_context, translate to wire shapes."""
from __future__ import annotations

import asyncio
import logging
import pathlib
from types import SimpleNamespace

import pytest

from adapters.inbound.shell_component.component import ContextGraphComponent, register
from host.daemon_runtime.context import ShellContext, ServiceEndpoints
from host.daemon_runtime.registry import Registry
from domain.ports.daemon.operations import OperationContext, OperationError, Principal
from domain.ports.daemon.shell import HealthStatus
from domain.agent_envelope import AgentEnvelope, EvidenceItem, CoverageReport, UnsupportedInclude
from domain.ports.agent_context import RecordReceipt, StatusReport, SkillNudge


# --- fakes -----------------------------------------------------------------
class FakeAgentContext:
    def __init__(self):
        self.calls = []

    def resolve(self, req):
        self.calls.append(("resolve", req))
        return AgentEnvelope(
            pot_id=req.pot_id, intent=req.intent or "feature",
            items=(EvidenceItem(include="prior_bugs", candidate_key="c1", score=0.9,
                                payload={"fact": "x"}, coverage_status="complete"),),
            coverage=(CoverageReport(include="prior_bugs", status="complete", candidate_pool=1),),
            unsupported_includes=(UnsupportedInclude(name="weird", reason="unknown"),),
            overall_confidence="high",
        )

    def search(self, req):
        self.calls.append(("search", req))
        return AgentEnvelope(pot_id=req.pot_id, intent="feature", items=(), coverage=())

    def record(self, req):
        self.calls.append(("record", req))
        return RecordReceipt(pot_id=req.pot_id, record_type=req.record_type, accepted=True,
                             record_id="rec_1", status="recorded", mutations_applied=2)

    def status(self, req):
        self.calls.append(("status", req))
        return StatusReport(pot_id=req.pot_id, profile="local", daemon_up=True,
                            active_pot="default", backend_ready=True,
                            skills=SkillNudge(agent="claude", missing=("a",)),
                            recommended_next_action="go")


def _fake_host():
    pots = SimpleNamespace(
        active_pot=lambda: SimpleNamespace(pot_id="pot_x", name="default"),
        list_pots=lambda: [SimpleNamespace(pot_id="pot_x", name="default")],
    )
    return SimpleNamespace(agent_context=FakeAgentContext(), pots=pots)


async def _started_component(host) -> ContextGraphComponent:
    comp = ContextGraphComponent()
    ctx = ShellContext(config={"deps": host}, data_dir=pathlib.Path("/tmp"),
                       logger=logging.getLogger("t"), endpoints=ServiceEndpoints())
    await comp.on_start(ctx)
    return comp


def _octx():
    return OperationContext(principal=Principal(name="local"), request_id="r1")


# --- tests -----------------------------------------------------------------
def test_register_adds_factory():
    reg = Registry()
    register(reg)
    assert "context_graph" in reg.names()
    assert reg.create("context_graph").name == "context_graph"


@pytest.mark.asyncio
async def test_component_lifecycle_and_ops_list():
    comp = await _started_component(_fake_host())
    assert comp.health() is HealthStatus.READY
    assert [o.name for o in comp.operations()] == [
        "context.resolve", "context.search", "context.record", "context.status",
    ]
    await comp.on_stop()
    assert comp.health() is HealthStatus.STOPPED


@pytest.mark.asyncio
async def test_resolve_delegates_and_translates():
    host = _fake_host()
    comp = await _started_component(host)
    op = next(o for o in comp.operations() if o.name == "context.resolve")
    inp = op.input_model(task="fix login", intent="debugging", include=["prior_bugs"])
    out = await op.handler(inp, _octx())
    assert host.agent_context.calls[0][0] == "resolve"
    assert out.pot_id == "pot_x" and out.intent == "debugging"
    assert out.overall_confidence == "high"
    assert out.items[0].include == "prior_bugs" and out.items[0].payload["fact"] == "x"
    assert out.coverage[0].status == "complete"
    assert out.unsupported_includes[0].name == "weird"


@pytest.mark.asyncio
async def test_search_delegates():
    host = _fake_host()
    comp = await _started_component(host)
    op = next(o for o in comp.operations() if o.name == "context.search")
    out = await op.handler(op.input_model(lookup="OrderService"), _octx())
    assert host.agent_context.calls[0][0] == "search"
    assert out.pot_id == "pot_x"


@pytest.mark.asyncio
async def test_record_delegates_and_translates():
    host = _fake_host()
    comp = await _started_component(host)
    op = next(o for o in comp.operations() if o.name == "context.record")
    out = await op.handler(op.input_model(type="fix", summary="patched"), _octx())
    assert host.agent_context.calls[0][0] == "record"
    assert out.record_id == "rec_1" and out.status == "recorded" and out.mutations_applied == 2


@pytest.mark.asyncio
async def test_status_delegates_and_translates():
    host = _fake_host()
    comp = await _started_component(host)
    op = next(o for o in comp.operations() if o.name == "context.status")
    out = await op.handler(op.input_model(), _octx())
    assert host.agent_context.calls[0][0] == "status"
    assert out.daemon_up is True and out.active_pot == "default" and out.backend_ready is True
    assert out.skills.agent == "claude" and out.skills.missing == ["a"]


@pytest.mark.asyncio
async def test_explicit_pot_id_is_resolved_by_name():
    host = _fake_host()
    comp = await _started_component(host)
    op = next(o for o in comp.operations() if o.name == "context.resolve")
    await op.handler(op.input_model(task="t", pot_id="default"), _octx())
    # the request the host saw carried the resolved id, not the name
    assert host.agent_context.calls[0][1].pot_id == "pot_x"


@pytest.mark.asyncio
async def test_no_active_pot_raises_operation_error():
    host = _fake_host()
    host.pots.active_pot = lambda: None
    comp = await _started_component(host)
    op = next(o for o in comp.operations() if o.name == "context.status")
    with pytest.raises(OperationError) as ei:
        await op.handler(op.input_model(), _octx())
    assert ei.value.code == "not_found"


@pytest.mark.asyncio
async def test_unwired_host_raises_on_op():
    comp = ContextGraphComponent()
    ctx = ShellContext(config={}, data_dir=pathlib.Path("/tmp"),
                       logger=logging.getLogger("t"), endpoints=ServiceEndpoints())
    await comp.on_start(ctx)  # no deps wired
    op = next(o for o in comp.operations() if o.name == "context.status")
    with pytest.raises(OperationError) as ei:
        await op.handler(op.input_model(), _octx())
    assert ei.value.code == "unavailable"
