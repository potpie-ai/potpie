from app.modules.intelligence.agents.chat_agents.adaptive_agent import AdaptiveAgent
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.classification_prompts import AgentType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator


class CodeGenAgent(ChatAgent):
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
            role="Code Generation Agent",
            goal="Generate precise, copy-paste ready code modifications that maintain project consistency and handle all dependencies",
            backstory="""
                    You are an expert code generation agent specialized in creating production-ready,
                    immediately usable code modifications. Your primary responsibilities include:
                    1. Analyzing existing codebase context and understanding dependencies
                    2. Planning code changes that maintain exact project patterns and style
                    3. Implementing changes with copy-paste ready output
                    4. Following existing code conventions exactly as shown in the input files
                    5. Never modifying string literals, escape characters, or formatting unless specifically requested

                    Key principles:
                    - Provide required new imports in a separate code block
                    - Output only the specific functions/classes being modified
                    - Never change existing string formats or escape characters
                    - Maintain exact indentation and spacing patterns from original code
                    - Include clear section markers for where code should be inserted/modified
                """,
            tasks=[
                TaskConfig(
                    description=code_gen_task_prompt,
                    expected_output="User-friendly, clearly structured code changes with comprehensive dependency analysis, implementation details for ALL impacted files, and complete verification steps",
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
                "fetch_file",
                "analyze_code_structure",
            ]
        )
        if self.llm_provider.is_current_model_supported_by_pydanticai(
            config_type="chat"
        ):
            return PydanticRagAgent(self.llm_provider, agent_config, tools)
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

    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution
    HOW TO GUIDE:

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

    Work to generate copy-paste ready code:

    Follow this structured approach:

    Context Analysis:
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

    Implementation Planning:
    - Plan changes that maintain exact formatting
    - Never modify existing patterns unless requested
    - Identify required new imports
    - Plan changes for ALL files identified in steps 1 and 2
    - Consider impact on dependent files
    - Ensure changes maintain dependency compatibility
    - CRITICAL: Create concrete changes for EVERY impacted file
    - Map all required database schema updates
    - Detail API changes and version impacts

    CRITICAL: If any file that is REQUIRED to propose changes is missing, stop and request the user to provide the file using "@filename" or "@functionname". NEVER create hypothetical files.


    Code Generation Format:
    Structure your response in this user-friendly format:

    📝 Overview
    -----------
    A 2-3 line summary of the changes to be made.

    🔍 Dependency Analysis
    --------------------
    • Primary Changes:
        - file1.py: [brief reason]
        - file2.py: [brief reason]

    • Required Dependency Updates:
        - dependent1.py: [specific changes needed]
        - dependent2.py: [specific changes needed]

    • Database Changes:
        - Schema updates
        - Migration requirements
        - Data validation changes

    📦 Changes by File
    ----------------
    [REPEAT THIS SECTION FOR EVERY IMPACTED FILE, INCLUDING DEPENDENCIES]

    ### 📄 [filename.py]

    **Purpose of Changes:**
    Brief explanation of what's being changed and why

    **Required Imports:**
    ```python
    from new.module import NewClass
    ```

    **Code Changes:**
    ```python
    def modified_function():
        # Your code here
        pass
    ```

    [IMPORTANT: Include ALL dependent files with their complete changes]

    ⚠️ Important Notes
    ----------------
    • Breaking Changes: [if any]
    • Required Manual Steps: [if any]
    • Testing Recommendations: [if any]
    • Database Migration Steps: [if any]

    🔄 Verification Steps
    ------------------
    1. [Step-by-step verification process]
    2. [Expected outcomes]
    3. [How to verify the changes work]
    4. [Database verification steps]
    5. [API testing steps]

    Important Response Rules:
    1. Use clear section emojis and headers for visual separation
    2. Keep each section concise but informative
    3. Use bullet points and numbering for better readability
    4. Include only relevant information in each section
    5. Use code blocks with language specification
    6. Highlight important warnings or notes
    7. Provide clear, actionable verification steps
    8. Keep formatting consistent across all files
    9. Use emojis sparingly and only for section headers
    10. Maintain a clean, organized structure throughout
    11. NEVER skip dependent file changes
    12. Always include database migration steps when relevant
    13. Detail API version impacts and migration paths

    Remember to:
    - Format code blocks for direct copy-paste
    - Highlight breaking changes prominently
    - Make location instructions crystal clear
    - Include all necessary context for each change
    - Keep the overall structure scannable and navigable
    - MUST provide concrete changes for ALL impacted files
    - Include specific database migration steps when needed
    - Detail API versioning requirements

    The output should be easy to:
    - Read in a chat interface
    - Copy-paste into an IDE
    - Understand at a glance
    - Navigate through multiple files
    - Use as a checklist for implementation
    - Execute database migrations
    - Manage API versioning
"""
