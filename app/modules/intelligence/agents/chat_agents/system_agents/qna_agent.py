from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent import (
    PydanticMultiAgent,
    AgentType as MultiAgentType,
)
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)


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

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="QNA Agent",
            goal="Answer queries of the repo in a detailed fashion",
            backstory="""
                    You are a highly efficient and intelligent RAG agent capable of querying complex knowledge graphs and refining the results to generate precise and comprehensive responses.
                    Navigate codebases incrementally: seed only minimal structure (repo root and top-level domains, with at most one sublevel for major surfaces) and expand specific branches on demand. Chain tools deliberately—load the guide, map the relevant subtree with get_code_file_structure(path=...), then pull file contents once scope is narrow.
                    Your tasks include:
                    1. Analyzing the user's query and formulating an effective strategy to extract relevant information from the code knowledge graph.
                    2. Executing the query with minimal iterations, ensuring accuracy and relevance.
                    3. Refining and enriching the initial results to provide a detailed and contextually appropriate response.
                    4. Maintaining traceability by including relevant citations and references in your output.
                    5. Including relevant citations in the response.
                """,
            tasks=[
                TaskConfig(
                    description=qna_task_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided code context and tool results",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "ask_knowledge_graph_queries",
                "get_nodes_from_tags",
                "get_code_file_structure",
                "webpage_extractor",
                "web_search_tool",
                "github_tool",
                "get_linear_issue",
                "update_linear_issue",
                "get_jira_issue",
                "search_jira_issues",
                "create_jira_issue",
                "update_jira_issue",
                "get_jira_project_details",
                "link_jira_issues",
                "add_jira_comment",
                "transition_jira_issue",
                "get_jira_projects",
                "get_jira_project_users",
                "get_confluence_spaces",
                "get_confluence_page",
                "search_confluence_pages",
                "get_confluence_space_pages",
                "create_confluence_page",
                "update_confluence_page",
                "add_confluence_comment",
                "fetch_file",
                "analyze_code_structure",
                "bash_command",
            ]
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
                # Create specialized delegate agents for codebase Q&A using available agent types
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
                }
                return PydanticMultiAgent(
                    self.llm_provider, agent_config, tools, None, delegate_agents
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
        ctx = await self._seed_top_level_structure(ctx)

        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code context of the node_ids in query:\n {code_results}"
            )

        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(await self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk

    async def _seed_top_level_structure(self, ctx: ChatContext) -> ChatContext:
        """Seed a minimal code map once so the agent can expand branches on demand."""
        if "Top-level code map" in ctx.additional_context:
            return ctx

        try:
            file_structure = await self.tools_provider.file_structure_tool.fetch_repo_structure(
                project_id=ctx.project_id, path=None, max_depth=2
            )
            formatted_structure = self._format_top_level_structure(
                file_structure, max_depth=2
            )

            if formatted_structure:
                prefix = "" if ctx.additional_context in ("", None) else "\n"
                ctx.additional_context += (
                    f"{prefix}Top-level code map (seeded once; expand with get_code_file_structure(path=...)):\n"
                    f"{formatted_structure}"
                )
        except Exception as exc:
            logger.warning(f"Failed to seed top-level structure: {exc}")

        return ctx

    def _format_top_level_structure(self, structure, max_depth: int = 1) -> str:
        """Keep only root entries and one sublevel to keep initial context small."""
        try:
            if isinstance(structure, str):
                lines = []
                for line in structure.splitlines():
                    stripped = line.lstrip(" ")
                    indent = len(line) - len(stripped)
                    depth = indent // 2
                    if depth <= max_depth:
                        lines.append(line)
                return "\n".join(lines).strip()

            if isinstance(structure, dict):
                nodes = structure.get("children", [])
                return "\n".join(
                    self._format_structure_nodes(nodes, depth=0, max_depth=max_depth)
                ).strip()

            if isinstance(structure, list):
                return "\n".join(
                    self._format_structure_nodes(structure, depth=0, max_depth=max_depth)
                ).strip()
        except Exception as exc:
            logger.warning(f"Failed to format top-level structure: {exc}")

        return ""

    def _format_structure_nodes(self, nodes, depth: int, max_depth: int):
        lines = []
        for node in sorted(nodes, key=lambda n: n.get("name", "")):
            name = node.get("name", "")
            if not name:
                continue
            lines.append(f"{'  ' * depth}{name}")
            children = node.get("children", [])
            if children and depth < max_depth:
                lines.extend(
                    self._format_structure_nodes(
                        children, depth=depth + 1, max_depth=max_depth
                    )
                )
        return lines


qna_task_prompt = """
    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution
    HOW TO GUIDE:

    IMPORTANT: Traverse the codebase incrementally
    - Seed only the topmost structure once: repo root and key domains, with at most one subdirectory per major surface (e.g., app/api, app/modules/intelligence, potpie-ui). Keep the initial map tiny.
    - When you need more detail, lazily expand a specific branch by calling get_code_file_structure(path=...) for that subtree. Explore depth-first as needed.
    - Do not request the entire tree upfront. Always decide the next branch to expand based on the user's question.

    TOOL CHAINING FLOW:
    1. Load this guide and identify the relevant surface area.
    2. Call get_code_file_structure(path=...) on the target subtree to map it; repeat depth-first for nested folders as you narrow focus.
    3. After the structure is clear, fetch content with analyze_code_structure or GetCodeFrom* tools, then FetchFile for broader context if needed.

    Additional navigation steps:
    1. Use websearch, docstrings, or README to understand the feature.
    2. Use AskKnowledgeGraphQueries to locate functionality or keywords; use FetchFile to pull code once scope is defined.
    3. Use GetcodefromProbableNodeIDs to fetch code for specific classes/functions; use analyze_code_structure to list class/function nodes in a file.
    4. Use GetcodeFromMultipleNodeIDs to fetch code for nodeIDs fetched from tools before.
    5. Use GetNodeNeighboursFromNodeIDs to fetch code referencing or referenced by the current node (code snippet).
    6. Use the above tools to assemble the full context about the code in question and how it ties together.
    7. When files are large, prefer targeted line ranges; FetchFile may fail on very large files—fall back to code analysis tools with start/end lines and a small buffer.
    8. Use imported/reference lookups to understand control flow and dependencies.

    Analyze and enrich results:
    - Evaluate relevance, identify gaps
    - Develop scoring mechanism
    - Retrieve code only if docstring insufficient

    Compose response:
    - Organize results logically
    - Include citations and references
    - Provide comprehensive, focused answer

    Final review:
    - Check coherence and relevance
    - Identify areas for improvement
    - Format the file paths as follows (only include relevant project details from file path):
        path: potpie/projects/username-reponame-branchname-userid/gymhero/models/training_plan.py
        output: gymhero/models/training_plan.py

    Note:

    - Use available tools in the correct order: structure first, then code
    - Use markdown for code snippets with language name in the code block like python or javascript
    - Prioritize "Get Code and docstring From Probable Node Name" tool for stacktraces or specific file/function mentions
    - Prioritize "Get Code File Structure" tool to get the nested file structure of a relevant subdirectory when deeper levels are not provided

    Ground your responses in provided code context and tool results. Use markdown for code snippets. Be concise and avoid repetition. If unsure, state it clearly. For debugging, unit testing, or unrelated code explanations, suggest specialized agents.
    Tailor your response based on question type:

    - New questions: Provide comprehensive answers
    - Follow-ups: Build on previous explanations from the chat history
    - Clarifications: Offer clear, concise explanations
    - Comments/feedback: Incorporate into your understanding

    Indicate when more information is needed. Use specific code references. Adapt to user's expertise level. Maintain a conversational tone and context from previous exchanges.
    Ask clarifying questions if needed. Offer follow-up suggestions to guide the conversation.
    Provide a comprehensive response with deep context, relevant file paths, include relevant code snippets wherever possible. Format it in markdown format.
"""
