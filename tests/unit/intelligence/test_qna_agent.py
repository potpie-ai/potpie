from types import SimpleNamespace

import pytest

import app.modules.intelligence.agents.chat_agents.system_agents.qna_agent as qna_module
from app.modules.intelligence.agents.chat_agents.system_agents.qna_agent import (
    QnAAgent,
    REPO_GUIDANCE_FILES,
)


pytestmark = pytest.mark.unit


class DummyToolsProvider:
    def __init__(self):
        self.requested = None
        self.exclude_embedding_tools = None

    def get_tools(self, requested, exclude_embedding_tools=False):
        self.requested = requested
        self.exclude_embedding_tools = exclude_embedding_tools
        return []


class DummyRagAgent:
    def __init__(self, llm_provider, agent_config, tools):
        self.llm_provider = llm_provider
        self.agent_config = agent_config
        self.tools = tools


def _fake_llm_provider():
    return SimpleNamespace(
        supports_pydantic=lambda _: False,
        chat_config=SimpleNamespace(model="test-model", capabilities=[]),
    )


def test_qna_prompt_requires_rg_and_avoids_kg_tools():
    prompt = qna_module.qna_task_prompt

    assert "Use rg (ripgrep), never grep" in prompt
    assert "ask_knowledge_graph_queries" not in prompt
    assert "get_code_from_probable_node_name" not in prompt
    assert "get_node_neighbours_from_node_id" not in prompt
    assert "get_code_from_multiple_node_ids" not in prompt
    assert "get_nodes_from_tags" not in prompt


def test_qna_agent_tool_list_excludes_kg_and_includes_bash(monkeypatch):
    monkeypatch.setattr(qna_module, "PydanticRagAgent", DummyRagAgent)

    tools_provider = DummyToolsProvider()
    agent = QnAAgent(
        llm_provider=_fake_llm_provider(),
        tools_provider=tools_provider,
        prompt_provider=SimpleNamespace(),
    )

    built = agent._build_agent()

    assert isinstance(built, DummyRagAgent)
    assert "bash_command" in tools_provider.requested
    assert "get_code_file_structure" in tools_provider.requested
    assert "fetch_file" in tools_provider.requested
    assert "fetch_files_batch" in tools_provider.requested
    assert "analyze_code_structure" in tools_provider.requested

    # KG tools must NOT be requested
    assert "ask_knowledge_graph_queries" not in tools_provider.requested
    assert "get_code_from_probable_node_name" not in tools_provider.requested
    assert "get_node_neighbours_from_node_id" not in tools_provider.requested
    assert "get_code_from_multiple_node_ids" not in tools_provider.requested
    assert "get_nodes_from_tags" not in tools_provider.requested

    # Web/todo tools must NOT be requested
    assert "webpage_extractor" not in tools_provider.requested
    assert "web_search_tool" not in tools_provider.requested
    assert "read_todos" not in tools_provider.requested
    assert "write_todos" not in tools_provider.requested
    assert "add_todo" not in tools_provider.requested
    assert "update_todo_status" not in tools_provider.requested
    assert "remove_todo" not in tools_provider.requested
    assert "add_subtask" not in tools_provider.requested
    assert "set_dependency" not in tools_provider.requested
    assert "get_available_tasks" not in tools_provider.requested
    assert "add_requirements" not in tools_provider.requested
    assert "get_requirements" not in tools_provider.requested


def test_qna_prompt_colgrep_first():
    prompt = qna_module.qna_task_prompt

    assert "search_colgrep" in prompt
    assert "ColGREP" in prompt
    assert "RETRIEVAL PIPELINE" in prompt
    assert "ColGREP is your PRIMARY discovery tool" in prompt


def test_qna_tool_list_includes_colgrep(monkeypatch):
    monkeypatch.setattr(qna_module, "PydanticRagAgent", DummyRagAgent)

    tools_provider = DummyToolsProvider()
    agent = QnAAgent(
        llm_provider=_fake_llm_provider(),
        tools_provider=tools_provider,
        prompt_provider=SimpleNamespace(),
    )

    agent._build_agent()

    assert "search_colgrep" in tools_provider.requested
    assert "check_colgrep_health" in tools_provider.requested


def test_qna_prompt_no_web_or_todo_tools():
    prompt = qna_module.qna_task_prompt

    assert "webpage_extractor" not in prompt
    assert "web_search_tool" not in prompt
    assert "read_todos" not in prompt
    assert "write_todos" not in prompt
    assert "add_todo" not in prompt
    assert "update_todo_status" not in prompt
    assert "remove_todo" not in prompt


def test_qna_prompt_conservative_delegation():
    """Verify the delegate backstory contains conservative delegation policy."""
    # The delegation policy lives in the _build_agent backstory for the
    # THINK_EXECUTE delegate.  We can inspect it indirectly through the
    # module-level prompt and the agent construction.  The prompt itself
    # is module-level qna_task_prompt but the delegation text lives in
    # the backstory string inside _build_agent.  We test the source code
    # by building the agent with multi-agent enabled and capturing the config.

    # Since the backstory is built inside _build_agent when multi-agent is
    # enabled, and we already verified ColGREP references in the prompt,
    # we directly search the source text for the delegation policy strings.
    import inspect

    source = inspect.getsource(QnAAgent._build_agent)
    assert "CONSERVATIVE DELEGATION" in source
    assert "ColGREP searches" in source


def test_qna_repo_guidance_files_defined():
    assert isinstance(REPO_GUIDANCE_FILES, frozenset)
    assert "AGENTS.md" in REPO_GUIDANCE_FILES
    assert "agents.md" in REPO_GUIDANCE_FILES
    assert "skills.md" in REPO_GUIDANCE_FILES
    assert "SKILLS.md" in REPO_GUIDANCE_FILES
    assert ".github/copilot-instructions.md" in REPO_GUIDANCE_FILES
    assert ".cursor/rules" in REPO_GUIDANCE_FILES


def test_qna_accepts_tool_resolver(monkeypatch):
    monkeypatch.setattr(qna_module, "PydanticRagAgent", DummyRagAgent)

    tools_provider = DummyToolsProvider()

    # Default: tool_resolver=None
    agent_default = QnAAgent(
        llm_provider=_fake_llm_provider(),
        tools_provider=tools_provider,
        prompt_provider=SimpleNamespace(),
    )
    assert agent_default.tool_resolver is None
    agent_default._build_agent()  # should not raise

    # Explicit mock tool_resolver
    mock_resolver = SimpleNamespace(resolve=lambda name: None)
    agent_with_resolver = QnAAgent(
        llm_provider=_fake_llm_provider(),
        tools_provider=tools_provider,
        prompt_provider=SimpleNamespace(),
        tool_resolver=mock_resolver,
    )
    assert agent_with_resolver.tool_resolver is mock_resolver
    agent_with_resolver._build_agent()  # should not raise


@pytest.mark.asyncio
async def test_qna_enrichment_called_before_build(monkeypatch):
    """Verify that run() calls _enriched_context before _build_agent."""
    monkeypatch.setattr(qna_module, "PydanticRagAgent", DummyRagAgent)

    call_order = []

    tools_provider = DummyToolsProvider()
    agent = QnAAgent(
        llm_provider=_fake_llm_provider(),
        tools_provider=tools_provider,
        prompt_provider=SimpleNamespace(),
    )

    original_enriched = agent._enriched_context
    original_build = agent._build_agent

    async def mock_enriched_context(ctx):
        call_order.append("enriched_context")
        # Return ctx unchanged (skip real enrichment which needs real services)
        return ctx

    def mock_build_agent(ctx=None):
        call_order.append("build_agent")
        # Return a dummy agent with an async run method
        async def dummy_run(ctx):
            return SimpleNamespace(response="dummy", citations=[])

        return SimpleNamespace(run=dummy_run)

    monkeypatch.setattr(agent, "_enriched_context", mock_enriched_context)
    monkeypatch.setattr(agent, "_build_agent", mock_build_agent)

    # Create a minimal ChatContext-like object
    ctx = SimpleNamespace(
        project_id="test",
        additional_context="",
        is_inferring=lambda: False,
    )

    await agent.run(ctx)

    assert call_order == ["enriched_context", "build_agent"]
