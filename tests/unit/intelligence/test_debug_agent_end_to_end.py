"""End-to-end integration tests — DebugAgent hypothesis creation.

These tests verify that the agent machinery (wrap_structured_tools →
RecordHypothesisInput validation → record_hypothesis → HypothesisStore)
works correctly in the complete tool-call pipeline.

Two levels:

A. Tool-pipeline tests
   Uses pydantic_ai's FunctionModel (no real LLM API) to drive the agent.
   FunctionModel emits controlled tool calls; we verify the hypothesis
   store is populated correctly after the run.

B. Prompt-format consistency tests
   Confirm that the prompt's record_hypothesis example shows list syntax
   rather than string syntax — the mismatch is what caused the bug in
   production traces 019e44e1... and 019e44dc...

Design rationale
----------------
Production traces 019e44e1... and 019e44dc... showed two consistency failures:

  1. Both agents called record_hypothesis with evidence/validation_plan as
     multi-line strings (matching the prompt example), but RecordHypothesisInput
     expects List[str]. This caused pydantic ValidationError → handle_exception
     returned "An internal error occurred." → hypothesis store stayed empty →
     the agent fell back to prose analysis, skipping Phases 5-7 entirely.

  2. The Gemini trace grew to 112 K tokens in Phase 3 before completing Phase 4,
     hitting the model's max_tokens limit mid-analysis.

These tests guard against regression of failure (1).
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Capture original sys.modules / env state *before* this file stubs them. This
# module imports app modules at collection time, so the stubs must be registered
# here (not in a fixture); the autouse fixture below restores the original state
# after this module's tests so the fake torch/sentence_transformers and test DB
# env don't leak into the rest of the suite.
# ---------------------------------------------------------------------------
_STUBBED_MODULE_NAMES = (
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch._jit_internal",
    "torch._sources",
    "torch._VF",
    "sentence_transformers",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}
_ORIGINAL_ENV = {key: os.environ.get(key) for key in ("POSTGRES_SERVER", "REDIS_URL")}

# ---------------------------------------------------------------------------
# Env vars required before any app import (same pattern as test_debug_agent_routing.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


# ---------------------------------------------------------------------------
# Stub out heavy ML packages that are pulled in through the import chain.
# Only stubs if the real package isn't already loaded in this environment.
# ---------------------------------------------------------------------------
def _register_stub(name: str, **attrs: object) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_torch_stub = _register_stub("torch")
_register_stub("torch.nn", functional=MagicMock())
_register_stub("torch.nn.functional")
_register_stub("torch.nn.modules")
_register_stub("torch._jit_internal")
_register_stub("torch._sources")
_register_stub("torch._VF")
_torch_stub._VF = MagicMock()  # type: ignore[attr-defined]
_torch_stub.functional = MagicMock()  # type: ignore[attr-defined]
_torch_stub.nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]
_st_stub = _register_stub("sentence_transformers")
_st_stub.SentenceTransformer = MagicMock(name="SentenceTransformer")  # type: ignore[attr-defined]


@pytest.fixture(autouse=True, scope="module")
def _restore_global_import_state():
    """Restore sys.modules / env after this module's tests so the fake torch and
    sentence_transformers stubs (and test DB env) don't poison later tests."""
    yield
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    for key, original in _ORIGINAL_ENV.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


# ---------------------------------------------------------------------------
# Shared imports (deferred until after stubs are in place)
# ---------------------------------------------------------------------------
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (  # noqa: E402
    wrap_structured_tools,
)
from app.modules.intelligence.tools.hypothesis_state_tool import (  # noqa: E402
    HypothesisStore,
    _hypothesis_store_ctx,
    create_hypothesis_state_tools,
    list_hypotheses,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_hypothesis_store():
    """Each test gets its own isolated HypothesisStore."""
    store = HypothesisStore(conversation_id="end-to-end-test")
    _hypothesis_store_ctx.set(store)
    yield store
    _hypothesis_store_ctx.set(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_function_model(tool_calls: list[dict], final_text: str) -> FunctionModel:
    """Return a FunctionModel that emits *tool_calls* sequentially, then *final_text*.

    tool_calls: list of dicts with keys 'name' and 'args' (dict).
    """
    state = {"index": 0}

    def model_fn(messages: list, info: object) -> ModelResponse:
        idx = state["index"]
        state["index"] += 1
        if idx < len(tool_calls):
            tc = tool_calls[idx]
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tc["name"], args=tc["args"])]
            )
        return ModelResponse(parts=[TextPart(content=final_text)])

    return FunctionModel(model_fn)


def _run_agent(model: FunctionModel, user_message: str) -> str:
    """Run a pydantic_ai Agent with the real hypothesis tools and return its output."""
    hypothesis_tools = create_hypothesis_state_tools()
    wrapped = wrap_structured_tools(hypothesis_tools)
    agent = Agent(
        model=model,
        tools=wrapped,
        system_prompt="You are a hypothesis-driven debugging agent.",
    )

    async def _run():
        result = await agent.run(user_message)
        return result.output

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# A1. Happy path — list evidence flows through to the store
# ---------------------------------------------------------------------------


def test_agent_creates_hypothesis_with_list_evidence():
    """The pipeline must populate the hypothesis store when the LLM passes list evidence.

    This is the 'works correctly' baseline. If this test fails, the tool
    registration or wrap_structured_tools machinery is broken.
    """
    model = _make_function_model(
        tool_calls=[
            {
                "name": "record_hypothesis",
                "args": {
                    "title": "Zero-length check missing in zipmapValidateIntegrity",
                    "status": "proposed",
                    "evidence": [
                        "ASan shows heap-buffer-overflow in zipmapNext at rdb.c:2408",
                        "Error message: Hash zipmap with big length (0)",
                    ],
                    "validation_plan": [
                        "Set breakpoint at zipmapValidateIntegrity",
                        "Verify length == 0 is not rejected by validation",
                    ],
                },
            }
        ],
        final_text=(
            "## Hypothesis 1: Zero-length check missing in zipmapValidateIntegrity\n\n"
            "### Status: proposed\n\n"
            "### Evidence\n\n"
            "- ASan shows heap-buffer-overflow in zipmapNext at rdb.c:2408\n\n"
            "---"
        ),
    )

    _run_agent(model, "AddressSanitizer: heap-buffer-overflow in zipmapNext")

    stored = list_hypotheses()["hypotheses"]
    assert len(stored) == 1, f"Expected 1 hypothesis in store, got {len(stored)}"
    assert stored[0]["id"] == "hyp_1"
    assert stored[0]["status"] == "proposed"
    assert len(stored[0]["evidence"]) == 2


def test_agent_creates_multiple_hypotheses_in_order():
    """The store must hold all hypotheses in creation order with sequential ids."""
    model = _make_function_model(
        tool_calls=[
            {
                "name": "record_hypothesis",
                "args": {
                    "title": "Hypothesis A: primary root cause",
                    "evidence": ["Frame rdb.c:2408 in zipmapNext"],
                    "validation_plan": ["Breakpoint at zipmapValidateIntegrity"],
                },
            },
            {
                "name": "record_hypothesis",
                "args": {
                    "title": "Hypothesis B: secondary candidate",
                    "evidence": ["Frame rdb.c:2401 in rdbLoadObject"],
                    "validation_plan": ["Breakpoint at rdbLoadObject"],
                },
            },
        ],
        final_text=(
            "## Hypothesis 1: Hypothesis A ...\n---\n"
            "## Hypothesis 2: Hypothesis B ...\n---"
        ),
    )

    _run_agent(model, "heap-buffer-overflow")

    stored = list_hypotheses()["hypotheses"]
    assert len(stored) == 2
    assert stored[0]["id"] == "hyp_1"
    assert stored[1]["id"] == "hyp_2"
    assert "primary" in stored[0]["title"]
    assert "secondary" in stored[1]["title"]


# ---------------------------------------------------------------------------
# A2. String-evidence path — the production bug from traces 019e44e1 / 019e44dc
# ---------------------------------------------------------------------------


def test_agent_creates_hypothesis_with_string_evidence():
    """The pipeline must NOT silently drop the hypothesis when the LLM passes
    evidence as a multi-line string instead of a list.

    Root-cause reproduction: both production traces showed the LLM following the
    prompt example and passing evidence as a string. Without schema coercion,
    RecordHypothesisInput raised ValidationError → handle_exception returned
    'An internal error occurred.' → hypothesis store stayed empty.

    After the fix (field_validator coercing str to List[str]), the hypothesis
    must land in the store.
    """
    model = _make_function_model(
        tool_calls=[
            {
                "name": "record_hypothesis",
                "args": {
                    "title": "Zero-length check missing in zipmapValidateIntegrity",
                    "status": "proposed",
                    "evidence": (
                        "- ASan shows heap-buffer-overflow in zipmapNext at rdb.c:2408\n"
                        "- Error message: Hash zipmap with big length (0)"
                    ),
                    "validation_plan": (
                        "- Set breakpoint at zipmapValidateIntegrity\n"
                        "- Verify length == 0 is not rejected"
                    ),
                },
            }
        ],
        final_text="## Hypothesis 1: ...\n---",
    )

    _run_agent(model, "heap-buffer-overflow in zipmapNext")

    stored = list_hypotheses()["hypotheses"]
    assert len(stored) == 1, (
        f"Expected 1 hypothesis in store after string-evidence call, got {len(stored)}. "
        "This indicates RecordHypothesisInput is still rejecting string evidence — "
        "the field_validator coercion fix is missing."
    )
    assert stored[0]["status"] == "proposed"
    # Coerced evidence must be a non-empty list, not a raw string
    assert isinstance(stored[0]["evidence"], list)
    assert len(stored[0]["evidence"]) >= 1


def test_agent_creates_hypothesis_with_mixed_format_evidence():
    """Validate that a mix of list items for one field and string for another both work."""
    model = _make_function_model(
        tool_calls=[
            {
                "name": "record_hypothesis",
                "args": {
                    "title": "Mixed format test",
                    "evidence": ["Proper list item"],  # list — no coercion needed
                    "validation_plan": "- Step A\n- Step B",  # string — needs coercion
                },
            }
        ],
        final_text="Done",
    )

    _run_agent(model, "some error")

    stored = list_hypotheses()["hypotheses"]
    assert len(stored) == 1
    assert stored[0]["evidence"] == ["Proper list item"]
    assert isinstance(stored[0]["validation_plan"], list)
    assert len(stored[0]["validation_plan"]) >= 1


# ---------------------------------------------------------------------------
# A3. Post-creation tool interactions — update_hypothesis_status + list_hypotheses
# ---------------------------------------------------------------------------


def test_agent_can_update_hypothesis_status_after_recording():
    """After record_hypothesis succeeds, update_hypothesis_status must find the record."""
    model = _make_function_model(
        tool_calls=[
            {
                "name": "record_hypothesis",
                "args": {
                    "title": "Status update test",
                    "evidence": ["Initial observation"],
                    "validation_plan": ["Step 1"],
                },
            },
            {
                "name": "update_hypothesis_status",
                "args": {"hypothesis_id": "hyp_1", "status": "debugging"},
            },
        ],
        final_text="### Status: debugging",
    )

    _run_agent(model, "testing status update")

    stored = list_hypotheses()["hypotheses"]
    assert len(stored) == 1
    assert stored[0]["status"] == "debugging"


def test_agent_list_hypotheses_returns_all_created():
    """list_hypotheses tool call must reflect all hypotheses in the store."""
    model = _make_function_model(
        tool_calls=[
            {
                "name": "record_hypothesis",
                "args": {"title": "H1", "evidence": ["e1"], "validation_plan": ["p1"]},
            },
            {
                "name": "record_hypothesis",
                "args": {"title": "H2", "evidence": ["e2"], "validation_plan": ["p2"]},
            },
            {"name": "list_hypotheses", "args": {}},
        ],
        final_text="Listed all hypotheses.",
    )

    _run_agent(model, "list all hypotheses")

    stored = list_hypotheses()["hypotheses"]
    assert len(stored) == 2
    titles = {h["title"] for h in stored}
    assert titles == {"H1", "H2"}


# ---------------------------------------------------------------------------
# B. Prompt-format consistency — guard against the prompt reverting to string example
# ---------------------------------------------------------------------------


def test_prompt_record_hypothesis_example_shows_list_evidence():
    """The prompt's record_hypothesis example must show evidence as a Python list,
    not as a string literal.

    When the example shows evidence="<string>", the LLM follows it and passes a
    string, which (without schema coercion) causes ValidationError.
    """
    from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent_prompt import (
        debug_task_prompt,
    )

    # Find the record_hypothesis code block in the prompt
    assert "record_hypothesis(" in debug_task_prompt, (
        "Prompt must contain a record_hypothesis() call example"
    )

    rh_start = debug_task_prompt.find("record_hypothesis(")
    rh_end = debug_task_prompt.find(")", rh_start) + 1
    # Extend to include multi-line block (find matching closing paren)
    depth = 0
    for i, ch in enumerate(debug_task_prompt[rh_start:], start=rh_start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                rh_end = i + 1
                break
    rh_block = debug_task_prompt[rh_start:rh_end]

    # The evidence argument must use list syntax [...] not string syntax "..."
    assert "evidence=[" in rh_block, (
        "The record_hypothesis() example in the prompt must use "
        "evidence=[...] (list syntax) rather than evidence='...' (string syntax). "
        "String syntax causes the LLM to pass a string, which fails pydantic "
        "validation and returns 'An internal error occurred.'\n\n"
        f"Found block:\n{rh_block}"
    )
    assert "validation_plan=[" in rh_block, (
        "The record_hypothesis() example in the prompt must use "
        "validation_plan=[...] (list syntax) rather than validation_plan='...' (string syntax).\n\n"
        f"Found block:\n{rh_block}"
    )


# ---------------------------------------------------------------------------
# C. PydanticDeepDebugAgent — streaming + Logfire instrumentation
# ---------------------------------------------------------------------------
#
# These tests mock `pydantic_deep.create_deep_agent` so we can drive the
# adapter without a real LLM. We synthesize the node/event stream that
# PydanticDeepDebugAgent.run_stream consumes and assert the shape of the
# yielded ChatAgentResponse / ToolCallResponse objects.
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Iterable, List as _List  # noqa: E402
from unittest.mock import patch  # noqa: E402

from pydantic_ai.messages import (  # noqa: E402
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)

from app.modules.intelligence.agents.chat_agent import (  # noqa: E402
    ChatContext,
    ToolCallEventType,
)
from app.modules.intelligence.agents.chat_agents.agent_config import (  # noqa: E402
    AgentConfig,
    TaskConfig,
)


# ---------------------------------------------------------------------------
# Helpers — fake pydantic-deep agent that synthesizes node events
# ---------------------------------------------------------------------------


@dataclass
class _ModelRequestNodeStub:
    """Marker node yielded by the fake run, treated as a model-request node."""

    events: _List[Any]


@dataclass
class _CallToolsNodeStub:
    """Marker node yielded by the fake run, treated as a call-tools node."""

    events: _List[Any]


@dataclass
class _EndNodeStub:
    """Marker node yielded by the fake run, treated as the end node."""


class _FakeAgentRun:
    """Mimics the async iterator returned by ``agent.iter().__aenter__()``."""

    def __init__(self, nodes: Iterable[Any]):
        self._nodes = list(nodes)
        self.ctx = MagicMock(name="run_ctx")

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for node in self._nodes:
            # Wire the per-node stream() helper used by the adapter
            events = getattr(node, "events", [])
            node.stream = lambda _ctx, events=events: _stream_cm(events)
            yield node


@asynccontextmanager
async def _stream_cm(events: _List[Any]):
    async def _gen():
        for ev in events:
            yield ev

    yield _gen()


def _make_fake_agent(nodes: Iterable[Any]):
    """Return a fake pydantic-deep Agent whose .iter() yields *nodes*."""
    fake_agent = MagicMock(name="FakeDeepAgent")

    @asynccontextmanager
    async def _iter_cm(*args: Any, **kwargs: Any):
        yield _FakeAgentRun(nodes)

    fake_agent.iter = lambda *args, **kwargs: _iter_cm(*args, **kwargs)
    # `run` is used by the non-stream path; default to AsyncMock-style coroutine.
    async def _run(*args: Any, **kwargs: Any):
        result = MagicMock()
        result.output = "fake-non-stream-output"
        return result

    fake_agent.run = _run
    return fake_agent


def _patch_node_type_checks():
    """Patch pydantic_ai.Agent.is_*_node so they recognize our marker stubs.

    Returns a context-manager helper that may be used with ``with``."""
    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    return patch.multiple(
        pydantic_deep_debug_agent.Agent,
        is_model_request_node=staticmethod(
            lambda node: isinstance(node, _ModelRequestNodeStub)
        ),
        is_call_tools_node=staticmethod(
            lambda node: isinstance(node, _CallToolsNodeStub)
        ),
        is_end_node=staticmethod(lambda node: isinstance(node, _EndNodeStub)),
    )


def _make_agent_config() -> AgentConfig:
    return AgentConfig(
        role="DebugAgent",
        goal="Diagnose failures.",
        backstory="A focused debugger.",
        tasks=[
            TaskConfig(
                description="Debug the user's issue.",
                expected_output="Markdown response",
            )
        ],
    )


def _make_chat_context(**overrides: Any) -> ChatContext:
    kwargs: dict[str, Any] = dict(
        project_id="proj-debug",
        project_name="potpie",
        curr_agent_id="debug_agent",
        history=[],
        query="my code is broken",
        user_id="user-123",
        conversation_id="conv-456",
        local_mode=False,
    )
    kwargs.update(overrides)
    return ChatContext(**kwargs)


def _build_debug_agent(tools=None):
    """Construct a PydanticDeepDebugAgent with mocks for llm/tools."""
    from app.modules.intelligence.agents.chat_agents.pydantic_deep_debug_agent import (
        PydanticDeepDebugAgent,
    )

    llm_provider = MagicMock(name="llm_provider")
    llm_provider.get_pydantic_model.return_value = MagicMock(name="pydantic_model")
    return PydanticDeepDebugAgent(
        llm_provider=llm_provider,
        config=_make_agent_config(),
        tools=tools or [],
    )


def _make_part_start_event(content: str) -> PartStartEvent:
    return PartStartEvent(index=0, part=TextPart(content=content))


def _make_part_delta_event(content_delta: str) -> PartDeltaEvent:
    return PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=content_delta))


def _make_thinking_start_event(content: str) -> PartStartEvent:
    return PartStartEvent(index=0, part=ThinkingPart(content=content))


def _make_thinking_delta_event(content_delta: str) -> PartDeltaEvent:
    return PartDeltaEvent(
        index=0, delta=ThinkingPartDelta(content_delta=content_delta)
    )


def _make_tool_call_event(name: str, call_id: str, args: dict) -> FunctionToolCallEvent:
    return FunctionToolCallEvent(
        part=ToolCallPart(tool_name=name, tool_call_id=call_id, args=args)
    )


def _make_tool_result_event(name: str, call_id: str, result: str) -> FunctionToolResultEvent:
    return FunctionToolResultEvent(
        result=ToolReturnPart(
            tool_name=name,
            tool_call_id=call_id,
            content=result,
        )
    )


async def _collect_stream(agent, ctx):
    out: list = []
    async for chunk in agent.run_stream(ctx):
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# C1. Smoke test for text streaming
# ---------------------------------------------------------------------------


def test_pydantic_deep_debug_agent_streams_text_deltas():
    """Text from PartStartEvent + PartDeltaEvent should reach the consumer.

    Validates the core streaming contract: each text event becomes a
    ChatAgentResponse with that text in ``response``."""
    agent = _build_debug_agent()

    fake_pd = _make_fake_agent(
        nodes=[
            _ModelRequestNodeStub(
                events=[
                    _make_part_start_event("Hello "),
                    _make_part_delta_event("world"),
                    _make_part_delta_event("!"),
                ]
            ),
            _EndNodeStub(),
        ]
    )

    captured: dict[str, Any] = {}

    def _capture_create_deep_agent(*args: Any, **kwargs: Any):
        captured["instrument"] = kwargs.get("instrument")
        return fake_pd

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", side_effect=_capture_create_deep_agent
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    text_chunks = [c.response for c in chunks if c.response]
    assert text_chunks == ["Hello ", "world", "!"], (
        f"Expected text deltas in order, got {text_chunks!r}"
    )
    # No tool call responses in this scenario
    assert all(c.tool_calls == [] for c in chunks)


# ---------------------------------------------------------------------------
# C1b. Reasoning (ThinkingPart) is wrapped in <think> tags so the webview
#      renders it in a collapsible think block (same path as the code agent).
# ---------------------------------------------------------------------------


def test_pydantic_deep_debug_agent_wraps_reasoning_in_think_tags():
    """ThinkingPart/ThinkingPartDelta events must be emitted as <think>...</think>
    content, with the actual answer text after the closing tag."""
    agent = _build_debug_agent()

    fake_pd = _make_fake_agent(
        nodes=[
            _ModelRequestNodeStub(
                events=[
                    _make_thinking_start_event("Let me reason"),
                    _make_thinking_delta_event(" about the bug"),
                    _make_part_start_event("The fix is X"),
                ]
            ),
            _EndNodeStub(),
        ]
    )

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    joined = "".join(c.response for c in chunks if c.response)
    assert "<think>" in joined and "</think>" in joined, (
        f"Reasoning must be wrapped in think tags, got {joined!r}"
    )
    # Reasoning content sits inside the think block; answer after the close tag.
    think_body = joined[joined.index("<think>") + len("<think>") : joined.index("</think>")]
    assert "Let me reason about the bug" in think_body
    after = joined[joined.index("</think>") + len("</think>") :]
    assert "The fix is X" in after


def test_pydantic_deep_debug_agent_closes_think_tag_when_no_text_follows():
    """If the model reasons then goes straight to a tool call (no text), the
    <think> block must still be closed so the webview doesn't swallow the rest."""
    agent = _build_debug_agent()

    fake_pd = _make_fake_agent(
        nodes=[
            _ModelRequestNodeStub(
                events=[_make_thinking_start_event("reasoning with no answer text")]
            ),
            _EndNodeStub(),
        ]
    )

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    joined = "".join(c.response for c in chunks if c.response)
    assert joined.count("<think>") == 1
    assert joined.count("</think>") == 1
    assert joined.strip().endswith("</think>")


# ---------------------------------------------------------------------------
# C2. Tool-call streaming — CALL + RESULT event shapes
# ---------------------------------------------------------------------------


def test_pydantic_deep_debug_agent_streams_tool_call_events():
    """FunctionToolCallEvent + FunctionToolResultEvent should produce
    ToolCallResponse entries with CALL and RESULT event types respectively."""
    agent = _build_debug_agent()

    fake_pd = _make_fake_agent(
        nodes=[
            _CallToolsNodeStub(
                events=[
                    _make_tool_call_event(
                        "fetch_file", "call-1", {"file_path": "src/app.py"}
                    ),
                    _make_tool_result_event(
                        "fetch_file", "call-1", "def main(): pass"
                    ),
                ]
            ),
            _EndNodeStub(),
        ]
    )

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    tool_calls = [tc for c in chunks for tc in c.tool_calls]
    assert len(tool_calls) == 2, (
        f"Expected two tool-call chunks (CALL + RESULT), got {len(tool_calls)}"
    )

    call_evt = tool_calls[0]
    assert call_evt.tool_name == "fetch_file"
    assert call_evt.call_id == "call-1"
    assert call_evt.event_type == ToolCallEventType.CALL.value

    result_evt = tool_calls[1]
    assert result_evt.tool_name == "fetch_file"
    assert result_evt.call_id == "call-1"
    assert result_evt.event_type == ToolCallEventType.RESULT.value


# ---------------------------------------------------------------------------
# C3. Logfire ``instrument`` flag is wired through to create_deep_agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instrument_flag", [True, False])
def test_pydantic_deep_debug_agent_passes_instrument_flag(monkeypatch, instrument_flag):
    """should_instrument_pydantic_ai() value must be forwarded to create_deep_agent."""
    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    monkeypatch.setattr(
        pydantic_deep_debug_agent,
        "should_instrument_pydantic_ai",
        lambda: instrument_flag,
    )

    agent = _build_debug_agent()
    captured: dict[str, Any] = {}

    fake_pd = _make_fake_agent(nodes=[_EndNodeStub()])

    def _capture(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return fake_pd

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", side_effect=_capture
    ), _patch_node_type_checks():
        asyncio.run(_collect_stream(agent, ctx))

    assert captured.get("instrument") is instrument_flag, (
        f"Expected instrument={instrument_flag!r} in create_deep_agent kwargs, "
        f"got {captured.get('instrument')!r}"
    )


# ---------------------------------------------------------------------------
# C4. Logfire failures are non-fatal — agent still produces output
# ---------------------------------------------------------------------------


def test_pydantic_deep_debug_agent_runs_when_logfire_trace_metadata_raises(monkeypatch):
    """If logfire_trace_metadata blows up on entry, run_stream must still yield."""
    from contextlib import contextmanager

    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    @contextmanager
    def _boom(**_kwargs):
        raise RuntimeError("logfire exploded")
        yield  # pragma: no cover — unreachable

    monkeypatch.setattr(
        pydantic_deep_debug_agent, "logfire_trace_metadata", _boom
    )

    agent = _build_debug_agent()
    fake_pd = _make_fake_agent(
        nodes=[
            _ModelRequestNodeStub(
                events=[_make_part_start_event("alive")],
            ),
            _EndNodeStub(),
        ]
    )

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    assert any(c.response == "alive" for c in chunks), (
        "run_stream should still yield agent output when logfire_trace_metadata fails"
    )


def test_pydantic_deep_debug_agent_runs_when_logfire_span_raises(monkeypatch):
    """If is_logfire_enabled() is True but logfire.span raises, run_stream must
    still yield output (span failure is non-fatal)."""
    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    monkeypatch.setattr(
        pydantic_deep_debug_agent, "is_logfire_enabled", lambda: True
    )

    fake_logfire = types.ModuleType("logfire")

    def _bad_span(*_a, **_k):
        raise RuntimeError("span exploded")

    fake_logfire.span = _bad_span  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)

    agent = _build_debug_agent()
    fake_pd = _make_fake_agent(
        nodes=[
            _ModelRequestNodeStub(events=[_make_part_start_event("still-alive")]),
            _EndNodeStub(),
        ]
    )

    ctx = _make_chat_context()

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    assert any(c.response == "still-alive" for c in chunks), (
        "run_stream should still yield agent output when logfire.span raises"
    )


# ---------------------------------------------------------------------------
# C5. Non-streaming run() also passes instrument and survives logfire failures
# ---------------------------------------------------------------------------


def test_pydantic_deep_debug_agent_run_returns_output(monkeypatch):
    """The synchronous run() path must return the agent's output and pass
    instrument= into create_deep_agent (parity with run_stream)."""
    captured: dict[str, Any] = {}

    fake_agent = MagicMock(name="FakeAgent")

    async def _agent_run(*args: Any, **kwargs: Any):
        result = MagicMock()
        result.output = "diagnosis: cosmic rays"
        return result

    fake_agent.run = _agent_run

    def _capture(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return fake_agent

    debug_agent = _build_debug_agent()
    ctx = _make_chat_context()

    with patch("pydantic_deep.create_deep_agent", side_effect=_capture):
        resp = asyncio.run(debug_agent.run(ctx))

    assert resp.response == "diagnosis: cosmic rays"
    assert "instrument" in captured, "instrument flag must reach create_deep_agent"


def test_pydantic_deep_debug_agent_run_survives_logfire_failure(monkeypatch):
    """run() should still produce a response when logfire_trace_metadata fails."""
    from contextlib import contextmanager

    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    @contextmanager
    def _boom(**_kwargs):
        raise RuntimeError("logfire exploded")
        yield  # pragma: no cover — unreachable

    monkeypatch.setattr(
        pydantic_deep_debug_agent, "logfire_trace_metadata", _boom
    )

    fake_agent = MagicMock(name="FakeAgent")

    async def _agent_run(*args: Any, **kwargs: Any):
        result = MagicMock()
        result.output = "ok"
        return result

    fake_agent.run = _agent_run

    debug_agent = _build_debug_agent()
    ctx = _make_chat_context()

    with patch("pydantic_deep.create_deep_agent", return_value=fake_agent):
        resp = asyncio.run(debug_agent.run(ctx))

    assert resp.response == "ok"


# ---------------------------------------------------------------------------
# C6. Logfire metadata keys are forwarded correctly
# ---------------------------------------------------------------------------


def _expected_metadata_from_ctx(ctx: ChatContext) -> dict[str, Any]:
    """Mirror _build_logfire_metadata in the agent: the precise set of keys the
    helper is contractually required to populate. Re-deriving it here lets the
    test assert exact-key parity (no extras, no omissions)."""
    return {
        "user_id": ctx.user_id,
        "conversation_id": ctx.conversation_id,
        "agent_id": ctx.curr_agent_id or "debug_agent",
        "project_id": ctx.project_id,
        "project_name": ctx.project_name,
        "repository": getattr(ctx, "repository", None),
        "branch": getattr(ctx, "branch", None),
        "local_mode": ctx.local_mode,
        "runtime": "pydantic_deep_debug_agent",
    }


def test_pydantic_deep_debug_agent_forwards_logfire_metadata_keys(monkeypatch):
    """run() must call logfire_trace_metadata with the exact ctx-derived kwargs.

    Validates the trace-baggage contract: every key in _build_logfire_metadata
    is forwarded with the value sourced from the ChatContext (or the documented
    default for agent_id)."""
    from contextlib import contextmanager

    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    captured: dict[str, Any] = {}

    @contextmanager
    def _fake_trace_metadata(**kwargs: Any):
        captured["kwargs"] = dict(kwargs)
        captured["call_count"] = captured.get("call_count", 0) + 1
        yield

    monkeypatch.setattr(
        pydantic_deep_debug_agent, "logfire_trace_metadata", _fake_trace_metadata
    )

    ctx = _make_chat_context(
        user_id="user-meta",
        conversation_id="conv-meta",
        curr_agent_id="debug_agent",
        project_id="proj-meta",
        project_name="potpie-meta",
        local_mode=True,
        repository="acme/widgets",
        branch="feature/x",
    )

    fake_agent = MagicMock(name="FakeAgent")

    async def _agent_run(*args: Any, **kwargs: Any):
        result = MagicMock()
        result.output = "metadata-run-ok"
        return result

    fake_agent.run = _agent_run

    debug_agent = _build_debug_agent()

    with patch("pydantic_deep.create_deep_agent", return_value=fake_agent):
        resp = asyncio.run(debug_agent.run(ctx))

    assert resp.response == "metadata-run-ok"
    assert captured.get("call_count") == 1, (
        f"Expected logfire_trace_metadata to be called once, "
        f"got {captured.get('call_count')!r}"
    )

    expected = _expected_metadata_from_ctx(ctx)
    forwarded = captured["kwargs"]
    assert set(forwarded.keys()) == set(expected.keys()), (
        f"Metadata key set mismatch.\nExpected: {sorted(expected.keys())!r}\n"
        f"Got:      {sorted(forwarded.keys())!r}"
    )
    for key, expected_value in expected.items():
        assert forwarded[key] == expected_value, (
            f"Metadata key {key!r} mismatch: "
            f"expected {expected_value!r}, got {forwarded[key]!r}"
        )


def test_pydantic_deep_debug_agent_stream_forwards_logfire_metadata_keys(monkeypatch):
    """run_stream must also call logfire_trace_metadata exactly once with the
    same ctx-derived kwargs as the non-streaming run()."""
    from contextlib import contextmanager

    from app.modules.intelligence.agents.chat_agents import pydantic_deep_debug_agent

    captured: dict[str, Any] = {}

    @contextmanager
    def _fake_trace_metadata(**kwargs: Any):
        captured["kwargs"] = dict(kwargs)
        captured["call_count"] = captured.get("call_count", 0) + 1
        yield

    monkeypatch.setattr(
        pydantic_deep_debug_agent, "logfire_trace_metadata", _fake_trace_metadata
    )

    ctx = _make_chat_context(
        user_id="user-stream",
        conversation_id="conv-stream",
        curr_agent_id="debug_agent",
        project_id="proj-stream",
        project_name="potpie-stream",
        local_mode=True,
        repository="acme/streamer",
        branch="release/y",
    )

    agent = _build_debug_agent()
    fake_pd = _make_fake_agent(
        nodes=[
            _ModelRequestNodeStub(events=[_make_part_start_event("trace-ok")]),
            _EndNodeStub(),
        ]
    )

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        chunks = asyncio.run(_collect_stream(agent, ctx))

    # Confirm the stream actually produced output (so the metadata cm was
    # entered and exited around real work).
    assert any(c.response == "trace-ok" for c in chunks)
    assert captured.get("call_count") == 1, (
        f"Expected logfire_trace_metadata to be called exactly once per stream, "
        f"got {captured.get('call_count')!r}"
    )

    expected = _expected_metadata_from_ctx(ctx)
    forwarded = captured["kwargs"]
    assert set(forwarded.keys()) == set(expected.keys()), (
        f"Metadata key set mismatch (stream).\n"
        f"Expected: {sorted(expected.keys())!r}\n"
        f"Got:      {sorted(forwarded.keys())!r}"
    )
    for key, expected_value in expected.items():
        assert forwarded[key] == expected_value, (
            f"Metadata key {key!r} mismatch (stream): "
            f"expected {expected_value!r}, got {forwarded[key]!r}"
        )


# ---------------------------------------------------------------------------
# C7. Cancellation exits cooperatively
# ---------------------------------------------------------------------------


class _BlockingModelRequestNode(_ModelRequestNodeStub):
    """Model-request node whose .stream() yields one event then blocks forever.

    Subclasses _ModelRequestNodeStub so the patched is_model_request_node sees
    it as a model-request node. Overrides .stream() with our blocking variant
    so _FakeAgentRun-style node.stream rebinding is not needed."""

    def __init__(self, first_event: Any, block_event: asyncio.Event):
        super().__init__(events=[first_event])
        self._first_event = first_event
        self._block_event = block_event

    def stream(self, _ctx):  # mirrors pydantic-ai node API
        @asynccontextmanager
        async def _cm():
            async def _gen():
                yield self._first_event
                # Block indefinitely until cancelled; never reached after that.
                await self._block_event.wait()
                # pragma: no cover — unreachable under cancellation

            yield _gen()

        return _cm()


class _BlockingFakeAgentRun:
    """Variant of _FakeAgentRun that yields nodes without overwriting stream()."""

    def __init__(self, nodes: Iterable[Any]):
        self._nodes = list(nodes)
        self.ctx = MagicMock(name="run_ctx")

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for node in self._nodes:
            yield node


def test_pydantic_deep_debug_agent_stream_cancels_cooperatively():
    """When the stream consumer cancels mid-iteration (via wait_for timeout),
    the underlying ``agent.iter(...)`` context manager must exit cleanly and
    no unexpected exception should escape.

    The fake's model-request node emits one event then blocks forever; the test
    consumes the first chunk under a generous wait_for timeout, then provokes
    cancellation on the next pull with a short timeout, then closes the
    generator. We assert the iter context manager's __aexit__ fired."""
    never_event = asyncio.Event()
    blocking_node = _BlockingModelRequestNode(
        first_event=_make_part_start_event("first-chunk"),
        block_event=never_event,
    )

    # Tracking iter cm: yields a custom run that does NOT overwrite stream().
    class _TrackingBlockingIterCM:
        def __init__(self):
            self.entered = False
            self.exited = False
            self.exit_exc_type: Any = None

        async def __aenter__(self):
            self.entered = True
            return _BlockingFakeAgentRun([blocking_node, _EndNodeStub()])

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            self.exit_exc_type = exc_type
            return False  # don't suppress

    iter_cm = _TrackingBlockingIterCM()

    fake_pd = MagicMock(name="FakeDeepAgent")
    fake_pd.iter = lambda *args, **kwargs: iter_cm

    agent = _build_debug_agent()
    ctx = _make_chat_context()

    async def _drive():
        gen = agent.run_stream(ctx)
        try:
            # Pull the first chunk (the PartStartEvent → ChatAgentResponse).
            first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
            assert first.response == "first-chunk", (
                f"Expected first chunk to be 'first-chunk', got {first.response!r}"
            )

            # Now try to pull the second chunk; the underlying generator is
            # blocked on never_event.wait(), so wait_for will TimeoutError.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(gen.__anext__(), timeout=0.1)
        finally:
            # Closing the generator must complete cleanly (no leaked
            # CancelledError, no unhandled exception escaping aclose()).
            await gen.aclose()

    with patch(
        "pydantic_deep.create_deep_agent", return_value=fake_pd
    ), _patch_node_type_checks():
        asyncio.run(_drive())

    # The async-with around agent.iter(...) must have been entered and exited.
    assert iter_cm.entered, "iter() context manager was never entered"
    assert iter_cm.exited, (
        "iter() context manager was not exited — cancellation cleanup leaked. "
        "PydanticDeepDebugAgent.run_stream must close `async with agent.iter(...)` "
        "when the consumer cancels."
    )
