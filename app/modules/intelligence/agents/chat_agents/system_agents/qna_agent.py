import asyncio
import os
from typing import AsyncGenerator, Optional

from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from application.use_cases.query_context import (
    get_change_history as ce_get_change_history,
    get_decisions as ce_get_decisions,
)
from dataclasses import asdict

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from app.modules.context_graph.wiring import build_container_for_session
from app.modules.context_graph.bundle_renderer import (
    prefetch_runtime_banner,
    render_intelligence_bundle,
)
from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent import (
    PydanticMultiAgent,
    AgentType as MultiAgentType,
)
from app.modules.intelligence.agents.chat_agents.multi_agent.agent_factory import (
    create_integration_agents,
)
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.utils.logger import setup_logger
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext

logger = setup_logger(__name__)


class QnAAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self, ctx: Optional[ChatContext] = None) -> ChatAgent:
        bundle = getattr(ctx, "context_intelligence_bundle", None) if ctx else None
        cov = (bundle or {}).get("coverage") or {} if isinstance(bundle, dict) else {}
        prefetch_complete = (
            isinstance(cov, dict)
            and str(cov.get("status") or "").upper() == "COMPLETE"
        )

        if bundle and prefetch_complete:
            goal = (
                "Answer questions using prefetched CONTEXT INTELLIGENCE first when coverage is COMPLETE; "
                "avoid redundant graph tools; use code-level tools only when source is needed."
            )
            backstory = """
                    You are a codebase Q&A specialist. When Additional Context includes CONTEXT INTELLIGENCE
                    with COMPLETE evidence coverage, you answer directly from that block — you do not
                    re-fetch the same PR/decision/history data via tools. You use code tools (fetch_file,
                    etc.) only to show implementation the user asks for. For genuinely multi-file
                    investigations not covered by prefetch, you may explore further with a minimal tool set.
                    You keep a conversational tone and cite evidence clearly.
                """
        else:
            goal = (
                "Provide comprehensive, well-structured answers to questions about the codebase by "
                "systematically exploring code, understanding context, and delivering thorough explanations "
                "grounded in actual code."
            )
            backstory = """
                    You are an expert codebase analyst and Q&A specialist with deep expertise in systematically exploring and understanding codebases. You excel at:
                    1. Structured question analysis - breaking down complex questions into manageable exploration tasks
                    2. Systematic code navigation - methodically traversing knowledge graphs, code structures, and relationships
                    3. Context building - assembling comprehensive understanding from multiple code locations and perspectives
                    4. Clear communication - presenting technical information in an organized, accessible manner
                    5. Thorough verification - ensuring answers are complete, accurate, and well-supported by code evidence

                    You use todo and requirements tools to track complex multi-step questions, ensuring no aspect is missed. You maintain a conversational tone while being methodical and thorough.
                """

        agent_config = AgentConfig(
            role="Codebase Q&A Specialist",
            goal=goal,
            backstory=backstory,
            tasks=[
                TaskConfig(
                    description=qna_task_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided code context and tool results, with clear structure, citations, and comprehensive explanations",
                )
            ],
        )

        # Exclude embedding-dependent tools during INFERRING status
        exclude_embedding_tools = ctx.is_inferring() if ctx else False
        if exclude_embedding_tools:
            logger.info(
                "Project is in INFERRING status - excluding embedding-dependent tools (ask_knowledge_graph_queries, get_nodes_from_tags)"
            )

        tools = self.tools_provider.get_tools(
            [
                "get_pot_context",
                "get_decisions",
                "get_pr_review_context",
                "get_pr_diff",
                "get_change_history",
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "ask_knowledge_graph_queries",
                "get_nodes_from_tags",
                "get_code_file_structure",
                "webpage_extractor",
                "web_search_tool",
                "fetch_file",
                "fetch_files_batch",
                "analyze_code_structure",
                "bash_command",
                "read_todos",
                "write_todos",
                "add_todo",
                "update_todo_status",
                "remove_todo",
                "add_subtask",
                "set_dependency",
                "get_available_tasks",
                "add_requirements",
                "get_requirements",
            ],
            exclude_embedding_tools=exclude_embedding_tools,
        )

        supports_pydantic = self.llm_provider.supports_pydantic("chat")
        should_use_multi = MultiAgentConfig.should_use_multi_agent("codebase_qna_agent")

        logger.info(
            f"QnAAgent: supports_pydantic={supports_pydantic}, should_use_multi_agent={should_use_multi}"
        )
        logger.info(f"Current model: {self.llm_provider.chat_config.model}")
        logger.info(f"Model capabilities: {self.llm_provider.chat_config.capabilities}")

        if supports_pydantic:
            if should_use_multi:
                logger.info("✅ Using PydanticMultiAgent (multi-agent system)")
                # Create specialized delegate agents for codebase Q&A: THINK_EXECUTE + integration agents
                integration_agents = create_integration_agents()
                delegate_agents = {
                    MultiAgentType.THINK_EXECUTE: AgentConfig(
                        role="Q&A Synthesis Specialist",
                        goal="Synthesize findings and provide comprehensive answers to codebase questions",
                        backstory="You are skilled at combining technical analysis with clear communication to provide comprehensive answers about codebases.",
                        tasks=[
                            TaskConfig(
                                description="Synthesize code analysis and location findings into comprehensive, well-structured answers",
                                expected_output="Clear, comprehensive answers with code examples, explanations, and relevant context",
                            )
                        ],
                        max_iter=12,
                    ),
                    **integration_agents,
                }
                return PydanticMultiAgent(
                    self.llm_provider,
                    agent_config,
                    tools,
                    None,
                    delegate_agents,
                    tools_provider=self.tools_provider,
                )
            else:
                logger.info("❌ Multi-agent disabled by config, using PydanticRagAgent")
                return PydanticRagAgent(self.llm_provider, agent_config, tools)
        else:
            logger.error(
                f"❌ Model '{self.llm_provider.chat_config.model}' does not support Pydantic - using fallback PydanticRagAgent"
            )
            return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        # New: context-engine powered resolve_context prefetch (feature-flagged).
        # When enabled, prefer a single bundled resolution over multiple context tool calls.
        if os.getenv("CONTEXT_INTELLIGENCE_ENABLED", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            try:
                # Build a context-engine container using the host DB session
                # (QnA tools already operate under the same DB session in runtime).
                # Note: ctx.user_id is available but ChatContext here doesn't pass DB; we rely on tools' DB elsewhere.
                # For now we resolve using project-level container and existing settings.
                from app.core.database import SessionLocal

                db = SessionLocal()
                try:
                    container = build_container_for_session(db)
                    if container.resolution_service is not None and container.settings.is_enabled():
                        from domain.intelligence_models import ContextResolutionRequest
                        from application.use_cases.resolve_context import resolve_context as ce_resolve_context

                        req = ContextResolutionRequest(
                            project_id=ctx.project_id,
                            query=ctx.query,
                            consumer_hint=ctx.curr_agent_id,
                            timeout_ms=4000,
                        )
                        bundle = await ce_resolve_context(container.resolution_service, req)
                        bundle_dict = asdict(bundle)
                        ctx.context_intelligence_bundle = bundle_dict
                        ctx.additional_context += prefetch_runtime_banner(bundle_dict)
                        ctx.additional_context += render_intelligence_bundle(bundle_dict)
                        # If we have prefetched context, skip the older graph prefetch below.
                        # (Code context + file structure still happen later.)
                        return await self._append_code_and_structure(ctx)
                finally:
                    db.close()
            except Exception:
                logger.exception("Failed resolve_context prefetch; falling back to legacy prefetch")

        # Temporary hard requirement: fetch context-graph signal first.
        # This ensures QnA starts from historical intent/decision context.
        try:
            pot_context_tool = self.tools_provider.tools.get("get_pot_context")
            if pot_context_tool:
                payload = {
                    "pot_id": ctx.project_id,
                    "query": ctx.query,
                    "limit": 8,
                    "node_labels": ["PullRequest", "Decision", "Issue", "Feature"],
                }
                if hasattr(pot_context_tool, "ainvoke"):
                    context_rows = await pot_context_tool.ainvoke(payload)
                else:
                    context_rows = pot_context_tool.invoke(payload)
                ctx.additional_context += (
                    f"\nPot context (via get_pot_context, pre-fetched):\n"
                    f"{context_rows}\n"
                )
        except Exception:
            logger.exception("Failed prefetching pot context for QnA agent")

        # Neo4j structural layer (PR-linked decisions, change history) — same DB Graphiti uses for
        # episodes, but not returned by semantic search alone. Without this, the model only sees
        # Graphiti snippets unless it happens to call get_decisions / get_change_history.
        try:
            cg_settings = PotpieContextEngineSettings()
            if cg_settings.is_enabled() and ctx.project_id:
                structural = Neo4jStructuralAdapter(cg_settings)

                def _structural_snapshot() -> tuple[list, list]:
                    dec = ce_get_decisions(
                        structural, ctx.project_id, limit=12
                    )
                    hist = ce_get_change_history(
                        structural,
                        ctx.project_id,
                        limit=6,
                    )
                    return dec, hist

                decisions_preview, change_hist_preview = await asyncio.to_thread(
                    _structural_snapshot
                )
                ctx.additional_context += (
                    "\nContext graph — structural (Neo4j) pre-fetch "
                    "(code-linked + PR review/conversation decisions; recent PR↔code rows):\n"
                    f"- get_decisions-style rows (limit 12): {decisions_preview}\n"
                    f"- get_change_history-style rows (limit 6): {change_hist_preview}\n"
                )
        except Exception:
            logger.exception("Failed prefetching structural context graph for QnA agent")

        if ctx.node_ids and len(ctx.node_ids) > 0:
            return await self._append_code_and_structure(ctx)

        return await self._append_code_and_structure(ctx)

    async def _append_code_and_structure(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code context of the node_ids in query:\n {code_results}"
            )

        file_structure = (
            await self.tools_provider.file_structure_tool.fetch_repo_structure(
                ctx.project_id
            )
        )
        ctx.additional_context += f"File Structure of the project:\n {file_structure}"

        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        enriched_ctx = await self._enriched_context(ctx)
        return await self._build_agent(enriched_ctx).run(enriched_ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        enriched_ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent(enriched_ctx).run_stream(enriched_ctx):
            yield chunk


qna_task_prompt = """
# Codebase Q&A Agent — Structured Answering Guide

---

## ★ STEP 0: Context Intelligence Protocol (HIGHEST PRIORITY — read FIRST)

Before doing ANYTHING else, check Additional Context for a block titled:
**`=== CONTEXT INTELLIGENCE (PREFETCHED) ===`**

This block is produced by the intelligence layer and contains pre-resolved evidence:
semantic hits, artifacts (PR metadata), change history, decisions, discussions, and ownership.

### 0a. If the PREFETCHED block exists

Read the **`>>> MANDATORY TOOL-CALL RULES <<<`** section at the bottom of that block.
Those rules are BINDING for this turn. Summary:

| Coverage status | What you MUST do |
|---|---|
| **COMPLETE** | Answer **directly from the prefetched evidence**. Do NOT call `get_pot_context`, `get_decisions`, `get_change_history`, `get_pr_review_context`, `get_pr_diff`, or `ask_knowledge_graph_queries`. You may call code-level tools (`fetch_file`, `get_code_from_probable_node_name`, `analyze_code_structure`, `get_code_file_structure`, `get_node_neighbours_from_node_id`) ONLY if you need actual source code to complete the answer. |
| **PARTIAL** | Use the evidence that IS available. Call tools ONLY for the specific missing families listed. Do NOT re-fetch families already present. |
| **UNKNOWN / absent** | Fall through to the standard exploration flow below. |

### 0b. Per-query-type fast-paths (when prefetched block exists)

These are the most common query patterns. When coverage is COMPLETE, follow the fast-path:

- **"What happened in PR #X?" / "Summarize PR #X" / "Why was PR #X merged?"**
  → Answer from `[Artifacts]` + `[Change history]` + `[Decisions]` + `[Discussions]`.
  → Zero tool calls needed.

- **"Which files had review discussion in PR #X?"**
  → Answer from `[Discussions]` entries (look at file_path fields).
  → Zero tool calls needed. If all entries have file_path=null, say "no file-level review threads were recorded."

- **"Who owns file Y?"**
  → Answer from `[Ownership]` section. If ownership data is present, cite it directly.
  → If no ownership entry exists for that file, say "no ownership data available" — do NOT call tools; ownership is not available from any tool.

- **"What PRs modified file Y?" / "Show change history for file Y"**
  → Answer from `[Change history]` entries with PR numbers.
  → Zero tool calls needed.

- **"Find anything related to Z" (broad semantic search)**
  → Start from `[Semantic hits]`. List and summarize them.
  → If the user wants to see *implementation* code for those hits, then (and only then) call code-level tools to fetch the relevant source files.
  → Do NOT start a 15-tool exploration. Semantic hits ARE the answer for "find anything related to".

- **"Where is X implemented?" / "Show me the code for X"**
  → Semantic hits provide pointers. Use `get_code_from_probable_node_name` or `fetch_file` to retrieve actual source code. This is one of the few cases where tool calls are appropriate even with COMPLETE coverage.

### 0c. If no PREFETCHED block exists

Fall through to the standard exploration flow (Step 1 onward). Call `get_pot_context` first.

---

## Step 1: Understand the Question

### 1a. Analyze the Question Type

- **What questions**: "What does X do?" → functionality, purpose
- **How questions**: "How does X work?" → implementation, flow
- **Where questions**: "Where is X defined?" → location, usage
- **Why questions**: "Why was X changed?" → rationale, decisions, PR context
- **Multi-part**: Break into components

### 1b. Extract Key Information

Identify entities, scope, and complexity. If the prefetched block already covers the scope, skip planning tools.

### 1c. Plan (only for complex, multi-step questions WITHOUT prefetched coverage)

For complex questions that require deep code exploration:
1. Call `add_requirements` to document what needs answering
2. Call `add_todo` for each exploration step

For simple questions or questions answerable from prefetched evidence: skip planning entirely.

---

## Step 2: Code Navigation (only when prefetched evidence is insufficient)

### 2a. Context tools (ONLY if no prefetched block or coverage is PARTIAL)

- `get_pot_context`: semantic search for project entities (Graphiti)
- `get_pr_review_context`: PR review threads
- `get_pr_diff`: file-level diffs for a PR
- `get_decisions`: decisions linked to code
- `get_change_history`: PRs that modified code

**REMINDER**: If the prefetched block has COMPLETE coverage, do NOT call any of the above.

0. **When there is no prefetched CONTEXT INTELLIGENCE block (or coverage is PARTIAL)**:
   - Call `get_pot_context` first for semantic graph context (`pot_id` = project id, `query`, `limit=8`, node labels including `PullRequest` and `Decision`).
   - Additional context may include structural (Neo4j) pre-fetch rows; treat those as ground truth alongside Graphiti search.

0b. **PR discussions and design rationale** (when the question is about *why*, reviewer debate, or decisions on a known PR number):
   - Call `get_pr_review_context` with `pot_id` and `pr_number` to load full review-thread text linked to that PR.
   - Call `get_pr_diff` with `pot_id` + `pr_number` (and optional `file_path`) for concrete file-level patch excerpts.
   - Call `get_decisions` / `get_change_history` with optional `file_path` / `function_name` when needed.

### 2b. Code-level tools (always available when source code is needed)

- `fetch_file` / `fetch_files_batch`: read file contents
- `get_code_from_probable_node_name`: locate a class/function by name
- `get_code_from_multiple_node_ids`: fetch code from known node IDs
- `analyze_code_structure`: list classes/functions in a file
- `get_code_file_structure`: directory layout
- `get_node_neighbours_from_node_id`: find callers/callees

### 2c. External tools (when internal code is not enough)

- `web_search_tool`: external domain knowledge
- `webpage_extractor`: read documentation pages

---

## Step 3: Synthesize and Answer

### 3a. Answer quality checklist

- Directly address every part of the question
- Cite evidence: file paths, PR numbers, line ranges
- Use code snippets (markdown code blocks with language tag)
- Structure with clear headings for multi-part answers
- Strip project-detail prefixes from file paths

### 3b. Format

```
## [Direct Answer / Summary]

## Details
### [Aspect 1]
### [Aspect 2]

## Evidence / Code
```

### 3c. File path formatting

Strip project prefixes. Show only: `app/models/user.py` (not `potpie/projects/.../app/models/user.py`).

---

## Step 4: Tool Efficiency Rules

These rules apply to EVERY turn:

1. **Minimum viable tool calls**: Use the fewest tools necessary to fully answer the question. Every tool call should add new information not already available.
2. **No duplicate fetches**: Never call a tool for data already present in Additional Context.
3. **Stop when sufficient**: Once you have enough evidence to answer, stop exploring and write the answer. Do not keep calling tools "for thoroughness" if the answer is already clear.
4. **Todo/requirements tools**: Use only for genuinely complex multi-step explorations (3+ code locations to visit). Skip for simple or prefetched-answerable questions.
5. **Code tools are for code**: If the question is about history, decisions, ownership, or PR rationale, code-level tools are usually unnecessary.

---

## Communication Style

- Conversational, natural tone
- Technically precise
- Adaptive to user expertise
- Include follow-up suggestions when appropriate

---

## Handling Different Turn Types

- **New question**: Full answer with evidence
- **Follow-up**: Build on prior context, fill gaps
- **Clarification request**: Focused, concise explanation
- **Feedback**: Incorporate and adjust

---

## Reminder

Your goal: **correct, well-structured, evidence-grounded answers with the minimum tool overhead**. Prefetched intelligence exists to make you faster — trust it when coverage is COMPLETE.
"""
