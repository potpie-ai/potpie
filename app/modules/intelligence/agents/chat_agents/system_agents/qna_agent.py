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
from app.modules.utils.logger import setup_logger

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

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="QNA Agent",
            goal="Answer queries of the repo in a detailed fashion",
            backstory="""
                    You are a highly efficient and intelligent RAG agent capable of querying complex knowledge graphs and refining the results to generate precise and comprehensive responses.
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
                "search_user_memories",
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
        return await self._build_agent().run(await self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


qna_task_prompt = """
    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution

    🧠 **YOU HAVE ACCESS TO PAST MEMORIES:**
    You have the `search_user_memories` tool available. This gives you access to user's past interactions, preferences, decisions, and any relevant context from previous conversations. Use this tool whenever you need additional context.

    HOW TO GUIDE:

    🧠 **STEP 0: SEARCH MEMORIES FOR CONTEXT:**
    - **Access memories proactively** using `search_user_memories` throughout your work
    - Query for: "coding preferences", "project architecture", "past decisions", "[specific topic]"
    - Memories help you provide personalized, context-aware answers aligned with user's style
    - Search whenever you need more context about user's preferences or past interactions

    IMPORATANT: steps on HOW TO traverse the codebase:
    1. You can use websearch, docstrings, readme to understand current feature/code you are working with better. Understand how to use current feature in context of codebase
    2. Use AskKnowledgeGraphQueries tool to understand where perticular feature or functionality resides or to fetch specific code related to some keywords. Fetch file structure to understand the codebase better, Use FetchFile tool to fetch code from a file
    3. Use GetcodefromProbableNodeIDs tool to fetch code for perticular class or function in a file, Use analyze_code_structure to get all the class/function/nodes in a file
    4. Use GetcodeFromMultipleNodeIDs to fetch code for nodeIDs fetched from tools before
    5. Use GetNodeNeighboursFromNodeIDs to fetch all the code referencing current code or code referenced in the current node (code snippet)
    6. Above tools and steps can help you figure out full context about the current code in question
    7. Figure out how all the code ties together to implement current functionality
    8. Fetch Dir structure of the repo and use fetch file tool to fetch entire files, if file is too big the tool will throw error, then use code analysis tool to target proper line numbers (feel free to use set startline and endline such that few extra context lines are also fetched, tool won't throw out of bounds exception and return lines if they exist)
    9. Use above mentioned tools to fetch imported code, referenced code, helper functions, classes etc to understand the control flow

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
