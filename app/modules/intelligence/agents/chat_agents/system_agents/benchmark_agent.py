from app.modules.intelligence.agents.chat_agents.adaptive_agent import AdaptiveAgent
from app.modules.intelligence.agents.chat_agents.pydantic_4 import (
    PydanticToolGraphAgent,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_complex_task import (
    PydanticGraphAgent,
)
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent4 import (
    PydanticMultiAgent,
)
from app.modules.intelligence.prompts.classification_prompts import AgentType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator


class GithubIssueFixerAgent(ChatAgent):
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
            role="Issue Fixer Code Diff Generation agent",
            goal="Generate patch diffs to fix the issue described by the user",
            backstory="""
                    You are an expert code generation agent specialized in creating production-ready,
                    immediately usable code modifications. Your primary responsibilities include:
                    1. Analyzing existing codebase context and understanding dependencies
                    2. Planning code changes that maintain exact project patterns and style
                    3. Implementing changes as patch diffs
                    4. Following existing code conventions exactly as shown in the input files
                    5. Never modifying string literals, escape characters, or formatting unless specifically requested
                    6. Don't rewrite patch files or create them yourselves. Always use the file changes tools (which updates or uses FileChangeManager), Use file changes tools to update the repo. Load a file, replace lines, verify the file with context and generate diffs from there
                    7. FileChangeManager needs to updated with all the changes and then patch needs to generated from the FileChangeManager
                    8. Try not to read entire files directly, use line numbers to get parts of the code. Try to limit input token usage
                """,
            tasks=[
                TaskConfig(
                    description=code_gen_task_prompt,
                    expected_output="Patch Diffs of the changes to fix the given issue",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "ask_knowledge_graph_queries",
                "get_code_file_structure",
                "fetch_file",
                "web_search_tool",
            ]
        )
        if self.llm_provider.is_current_model_supported_by_pydanticai(
            config_type="chat"
        ):
            return PydanticMultiAgent(self.llm_provider, agent_config, tools)
        else:
            return AdaptiveAgent(
                llm_provider=self.llm_provider,
                prompt_provider=self.prompt_provider,
                rag_agent=CrewAIAgent(self.llm_provider, agent_config, tools),
                agent_type=AgentType.QNA,
            )

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
        return await self._build_agent().run(await self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


code_gen_task_prompt = """
    Follow this structured approach:

    1. Query Analysis:
    - Identify ALL file names or function names mentioned in the query
    - For files without node_ids, use get_code_from_probable_node_name tool
    - Example: "Update file1.py and config.py" -> fetch config.py and file1.py using tool if you dont already have their code
    - Look for words that could be file names or function names based on the query (e.g., requirements, utils, update document etc.)
    - Identify any data storage or schema changes that might affect multiple files

    2. Dependency Analysis:
    - Use get_node_neighbours tool on EACH function or file to be modified (works best with function names)
    - Analyze import relationships and dependencies EXHAUSTIVELY
    - Identify ALL files that import the modified files
    - Identify ALL files that interact with the modified functionality
    - Map the complete chain of dependencies:
    * Direct importers
    * Interface implementations
    * Shared data structures
    * Database interactions
    * API consumers
    - Document required changes in ALL dependent files
    - Flag any file that touches the modified functionality, even if changes seem minor

    3. Context Analysis:
    - Review existing code precisely to maintain standard formatting
    - Note exact indentation patterns
    - Identify string literal formats
    - Review import organization patterns
    - Ensure ALL required files are fetched before proceeding
    - Check dependency compatibility
    - Analyze database schemas and interactions
    - Review API contracts and interfaces
    - IF NO SPECIFIC FILES ARE FOUND:
    * FIRST Use get_file_structure tool to get the file structure of the project and get any relevant file context
    * THEN IF STILL NO SPECIFIC FILES ARE FOUND, use get_nodes_from_tags tool to search by relevant tags

    4. Implementation Planning:
    - Plan changes that maintain exact formatting
    - Never modify existing patterns unless requested
    - Identify required new imports
    - Plan changes for ALL files identified in steps 1 and 2
    - Consider impact on dependent files
    - Ensure changes maintain dependency compatibility
    - CRITICAL: Create concrete changes for EVERY impacted file
    - Map all required database schema updates
    - Detail API changes and version impacts
    
    5. Output generation:
    - Use file changes tools to update the repo. Load a file, replace lines, verify the file with context and generate diffs from there
    - Load the files that need to changed
    - Update the file using replace lines tool
    - Use read file in FileContentManager tool to confirm changes
    - You can reset the file using load file incase FileContentManager file state has messed up and hard to repair
    - Use search file to get proper line number in a larger file to get exact lines
    
    
    IMPORTANT: Reuse existing helpers in the project, explore the project and helper files and reuse the helpers and already existing functions/classes etc 

"""
