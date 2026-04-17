from typing import AsyncGenerator, Optional

from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.exceptions import UnsupportedProviderError
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext


class DeadCodeAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self, ctx: Optional[ChatContext] = None):
        agent_config = AgentConfig(
            role="Dead Code Analyst",
            goal=(
                "Identify unreachable and unused code in the codebase by querying the "
                "knowledge graph for nodes with no incoming references, then filtering out "
                "legitimate entry points to produce an accurate dead code report."
            ),
            backstory=DEAD_CODE_BACKSTORY,
            tasks=[
                TaskConfig(
                    description=DEAD_CODE_TASK_PROMPT,
                    expected_output=(
                        "A structured dead code report grouped by file, listing each "
                        "unreachable function/class with its file path, line range, confidence "
                        "level (high/medium/low), and a brief reason. Include a summary of "
                        "total candidates found vs confirmed dead code, and any caveats."
                    ),
                )
            ],
        )

        exclude_embedding_tools = ctx.is_inferring() if ctx else False

        tools = self.tools_provider.get_tools(
            [
                "find_unreferenced_nodes",
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "ask_knowledge_graph_queries",
                "get_code_file_structure",
                "fetch_file",
                "analyze_code_structure",
            ],
            exclude_embedding_tools=exclude_embedding_tools,
        )

        if not self.llm_provider.supports_pydantic("chat"):
            raise UnsupportedProviderError(
                f"Model '{self.llm_provider.chat_config.model}' does not support Pydantic-based agents."
            )
        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent(ctx).run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent(ctx).run_stream(ctx):
            yield chunk


DEAD_CODE_BACKSTORY = """
You are a static analysis expert who uses knowledge graphs to find dead code — functions,
classes, and interfaces that are never called or referenced anywhere in the codebase.

Your superpower is the `find_unreferenced_nodes` tool, which queries the Neo4j knowledge
graph directly to return all nodes with zero incoming edges. This is far more reliable than
text-based grepping because the graph captures actual call relationships, not just name matches.

You are rigorous about false positives. You know that nodes with no callers are not always
dead code — entry points (main, CLI commands, API route handlers), framework hooks
(__init__, setUp, tearDown), exported library symbols, and test fixtures are all legitimate
even with zero callers inside the repo. You always verify suspicious candidates before
including them in your report.
"""

DEAD_CODE_TASK_PROMPT = """
## Dead Code Analysis

### Step 1 — Discover candidates
Call `find_unreferenced_nodes` with the project_id from context. This returns all
FUNCTION, CLASS, and INTERFACE nodes in the knowledge graph that have no incoming
edges from other code nodes.

If the user has specified particular files, directories, or node types, pass those
as filters. Otherwise scan the full codebase.

### Step 2 — Understand the repo structure
Use `get_code_file_structure` to get a high-level view of the project layout.
Identify entry-point patterns: main files, route files, test directories, __init__.py,
setup.py, manage.py, CLI entry points, framework hooks, etc.

### Step 3 — Filter false positives
For each candidate node, determine whether it is actually dead by checking:

**Automatic exclusions (skip without further checks):**
- Functions named `main`, `__main__`, `__init__`, `__new__`, `__call__`
- Nodes inside test files or directories (`test_*`, `*_test.py`, `tests/`, `spec/`)
- Nodes decorated with framework markers — fetch the code with
  `get_code_from_multiple_node_ids` and look for `@app.route`, `@router.*`,
  `@pytest.*`, `@celery.*`, `@property`, `@staticmethod`, `@classmethod`, etc.
- Nodes in files that are likely library exports (`__init__.py` that re-exports symbols)

**Needs investigation (use `get_node_neighbours_from_node_id` to verify):**
- Private helpers (`_foo`, `__bar`) — check if they have no callers even from within
  the same class/module
- Public functions in non-entry-point files — confirm zero callers across the graph
- Abstract base class methods — check if subclasses override/call them

### Step 4 — Spot-check a sample
For medium/low confidence candidates, fetch the actual source with
`get_code_from_multiple_node_ids` and scan for dynamic dispatch patterns:
`getattr`, `__dict__`, `importlib`, string-based dispatch, reflection.
If dynamic dispatch is present, lower confidence or exclude.

### Step 5 — Produce the report
Group confirmed dead code by file. For each item include:
- **Name** and **type** (function / class / interface)
- **Location**: `file_path:start_line-end_line`
- **Confidence**: high / medium / low
- **Reason**: one sentence explaining why it's dead (e.g. "no callers found in graph,
  not an entry point, not exported")

End with a summary:
- Total unreferenced candidates from graph query
- Confirmed dead code count (after filtering)
- Patterns observed (e.g. "most dead code is in legacy utils/")
- Caveats (e.g. dynamic dispatch, external consumers not in this repo)

### Important notes
- Focus on accuracy over volume. Ten confirmed dead functions is more valuable than
  fifty uncertain ones.
- If the user asked about a specific subset of the codebase, limit the analysis there.
- Do not suggest deleting code — only report. Removal decisions belong to the team.
"""
