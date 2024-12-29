import asyncio
import os
from contextlib import redirect_stdout
from typing import Any, AsyncGenerator, Dict, List

import aiofiles
from crewai import Agent, Crew, Process, Task

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.intelligence.tools.code_query_tools.get_code_file_structure import (
    get_code_file_structure_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    get_node_neighbours_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    GetCodeFromMultipleNodeIdsTool,
    get_code_from_multiple_node_ids_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)


class CodeGenerationAgent:
    def __init__(self, sql_db, llm, mini_llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iter = os.getenv("MAX_ITER", 15)
        self.sql_db = sql_db
        self.get_code_from_multiple_node_ids = get_code_from_multiple_node_ids_tool(
            sql_db, user_id
        )
        self.get_node_neighbours = get_node_neighbours_from_node_id_tool(sql_db)
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.query_knowledge_graph = get_ask_knowledge_graph_queries_tool(
            sql_db, user_id
        )
        self.get_nodes_from_tags = get_nodes_from_tags_tool(sql_db, user_id)
        self.get_file_structure = get_code_file_structure_tool(sql_db)
        self.llm = llm
        self.mini_llm = mini_llm
        self.user_id = user_id

    async def create_agents(self):
        # [Previous create_agents code remains the same until the task description]
        code_generator = Agent(
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
                - Include clear section markers for where code should be inserted/modified            """,
            tools=[
                self.get_code_from_multiple_node_ids,
                self.get_node_neighbours,
                self.get_code_from_probable_node_name,
                self.query_knowledge_graph,
                self.get_nodes_from_tags,
                self.get_file_structure,
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iter,
        )

        return code_generator

    async def create_tasks(
        self,
        query: str,
        project_id: str,
        history: str,
        node_ids: List[NodeContext],
        code_results: List[Dict[str, Any]],
        code_generator,
    ):
        code_generation_task = Task(
            description=f"""
            Work within {self.max_iter} iterations to generate copy-paste ready code based on:
            - Query: {query}
            - Project ID: {project_id}
            - History: {history}
            - Target Node IDs: {[node.model_dump() for node in node_ids]}
            - Existing Code Context: {code_results}

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

            CRITICAL: If any file that is REQUIRED to propose changes is missing, stop and request the user to provide the file using "@filename" or "@functionname". NEVER create hypothetical files.


            5. Code Generation Format:
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
            """,
            expected_output="User-friendly, clearly structured code changes with comprehensive dependency analysis, implementation details for ALL impacted files, and complete verification steps",
            agent=code_generator,
        )

        return code_generation_task

    async def run(
        self,
        query: str,
        project_id: str,
        history: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        code_results = []
        if len(node_ids) > 0:
            code_results = await GetCodeFromMultipleNodeIdsTool(
                self.sql_db, self.user_id
            ).run_multiple(project_id, [node.node_id for node in node_ids])

        code_generator = await self.create_agents()
        generation_task = await self.create_tasks(
            query,
            project_id,
            history,
            node_ids,
            code_results,
            code_generator,
        )

        read_fd, write_fd = os.pipe()

        async def kickoff():
            with os.fdopen(write_fd, "w", buffering=1) as write_file:
                with redirect_stdout(write_file):
                    crew = Crew(
                        agents=[code_generator],
                        tasks=[generation_task],
                        process=Process.sequential,
                        verbose=True,
                    )
                    await crew.kickoff_async()

        asyncio.create_task(kickoff())

        # Stream the output
        final_answer_streaming = False
        async with aiofiles.open(read_fd, mode="r", encoding='utf-8') as read_file:
            async for line in read_file:
                if not line:
                    break
                if final_answer_streaming:
                    if line.endswith("\x1b[00m\n"):
                        yield line[:-6]
                    else:
                        yield line
                if "## Final Answer:" in line:
                    final_answer_streaming = True


async def kickoff_code_generation_crew(
    query: str,
    project_id: str,
    history: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> AsyncGenerator[str, None]:
    provider_service = ProviderService(sql_db, user_id)
    crew_ai_mini_llm = await provider_service.get_small_llm(agent_type=AgentType.CREWAI)
    crew_ai_llm = await provider_service.get_large_llm(agent_type=AgentType.CREWAI)
    code_gen_agent = CodeGenerationAgent(sql_db, crew_ai_llm, crew_ai_mini_llm, user_id)
    async for chunk in code_gen_agent.run(query, project_id, history, node_ids):
        yield chunk
