from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator


class DebugAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
    ):
        self.tools_provider = tools_provider

        agent_config = AgentConfig(
            role="Context curation agent",
            goal="Handle querying the knowledge graph and refining the results to provide accurate and contextually rich responses.",
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
                    description=code_gen_task_prompt,
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
                "github_tool",
            ]
        )

        self.rag_agent = CrewAIAgent(llm_provider, agent_config, tools)
        if llm_provider.is_current_model_supported_by_pydanticai():
            self.rag_agent = PydanticRagAgent(llm_provider, agent_config, tools)

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code referred to in the query:\n {code_results}\n"
            )
        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self.rag_agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self.rag_agent.run_stream(ctx):
            yield chunk


code_gen_task_prompt = """
    1. Analyze project structure:

    - Identify key directories, files, and modules
    - Guide search strategy and provide context
    - For directories of interest that show "└── ...", use "Get Code File Structure" tool with the directory path to reveal nested files
    - Only after getting complete file paths, use "Get Code and docstring From Probable Node Name" tool
    - Locate relevant files or subdirectory path


    Directory traversal strategy:

    - Start with high-level file structure analysis
    - When encountering a directory with hidden contents (indicated by "└── ..."):
        a. First: Use "Get Code File Structure" tool with the directory path
        b. Then: From the returned structure, identify relevant files
        c. Finally: Use "Get Code and docstring From Probable Node Name" tool with the complete file paths
    - Subdirectories with hidden nested files are followed by "│   │   │          └── ..."


    2. Initial context retrieval:
        - Analyze provided Code Results for user node ids
        - If code results are not relevant move to next step`

    3. Knowledge graph query (if needed):
        - Transform query for knowledge graph tool
        - Execute query and analyze results

    Additional context retrieval (if needed):

    - For each relevant directory with hidden contents:
        a. FIRST: Call "Get Code File Structure" tool with directory path
        b. THEN: From returned structure, extract complete file paths
        c. THEN: For each relevant file, call "Get Code and docstring From Probable Node Name" tool
    - Never call "Get Code and docstring From Probable Node Name" tool with directory paths
    - Always ensure you have complete file paths before using the probable node tool
    - Extract hidden file names from the file structure subdirectories that seem relevant
    - Extract probable node names. Nodes can be files or functions/classes. But not directories.


    5. Use "Get Nodes from Tags" tool as last resort only if absolutely necessary

    6. Analyze and enrich results:
        - Evaluate relevance, identify gaps
        - Develop scoring mechanism
        - Retrieve code only if docstring insufficient

    7. Compose response:
        - Organize results logically
        - Include citations and references
        - Provide comprehensive, focused answer

    8. Final review:
        - Check coherence and relevance
        - Identify areas for improvement
        - Format the file paths as follows (only include relevant project details from file path):
            path: potpie/projects/username-reponame-branchname-userid/gymhero/models/training_plan.py
            output: gymhero/models/training_plan.py

    Note:

    - Always traverse directories before attempting to access files
    - Never skip the directory structure retrieval step
    - Use available tools in the correct order: structure first, then code
    - Use markdown for code snippets with language name in the code block like python or javascript
    - Prioritize "Get Code and docstring From Probable Node Name" tool for stacktraces or specific file/function mentions
    - Prioritize "Get Code File Structure" tool to get the nested file structure of a relevant subdirectory when deeper levels are not provided
    - Use available tools as directed
    - Proceed to next step if insufficient information found

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
