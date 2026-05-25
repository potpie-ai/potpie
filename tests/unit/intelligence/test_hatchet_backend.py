"""Unit tests for the Hatchet agent enqueue side (no live Hatchet server)."""

import pytest

from app.modules.intelligence.agents.runtime import hatchet_backend as hb


def test_agent_run_input_round_trips_core_fields():
    payload = hb.AgentRunInput(
        conversation_id="c1",
        run_id="r1",
        user_id="u1",
        query="hello",
        agent_id="debugging_agent",
    ).model_dump()
    assert payload["conversation_id"] == "c1"
    assert payload["query"] == "hello"
    assert payload["agent_id"] == "debugging_agent"
    assert payload["operation"] == hb.AGENT_RUN_OPERATION_MESSAGE
    # optional fields default sanely
    assert payload["node_ids"] is None
    assert payload["attachment_ids"] == []
    assert payload["local_mode"] is False
    assert payload["tunnel_url"] is None


class _FakeEvent:
    def __init__(self):
        self.pushed = []

    def push(self, key, payload):
        self.pushed.append((key, payload))


class _FakeClient:
    def __init__(self):
        self.event = _FakeEvent()


def test_enqueue_agent_run_pushes_event_with_payload():
    client = _FakeClient()
    inp = hb.AgentRunInput(
        conversation_id="c1", run_id="r1", user_id="u1", query="hi", agent_id="debugging_agent"
    )
    hb.enqueue_agent_run(inp, client=client)
    assert len(client.event.pushed) == 1
    key, payload = client.event.pushed[0]
    assert key == hb.EVENT_AGENT_RUN
    assert payload["conversation_id"] == "c1"
    assert payload["agent_id"] == "debugging_agent"


def test_enqueue_regenerate_run_pushes_same_agent_event_with_operation():
    client = _FakeClient()
    inp = hb.AgentRunInput(
        conversation_id="c1",
        run_id="r1",
        user_id="u1",
        agent_id="debugging_agent",
        operation=hb.AGENT_RUN_OPERATION_REGENERATE,
        node_ids=[{"node_id": "n1", "name": "Node"}],
    )
    hb.enqueue_agent_run(inp, client=client)
    key, payload = client.event.pushed[0]
    assert key == hb.EVENT_AGENT_RUN
    assert payload["operation"] == hb.AGENT_RUN_OPERATION_REGENERATE
    assert payload["query"] == ""
    assert payload["node_ids"] == [{"node_id": "n1", "name": "Node"}]


def test_enqueue_agent_run_raises_when_push_fails():
    class _BoomEvent:
        def push(self, key, payload):
            raise RuntimeError("hatchet unreachable")

    class _BoomClient:
        event = _BoomEvent()

    inp = hb.AgentRunInput(
        conversation_id="c1", run_id="r1", user_id="u1", query="hi", agent_id="x"
    )
    with pytest.raises(RuntimeError):
        hb.enqueue_agent_run(inp, client=_BoomClient())
