import asyncio
import os
from contextlib import redirect_stdout
from typing import Any, AsyncGenerator, Dict, List

import agentops
import aiofiles
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.code_provider.code_provider_service import CodeProviderService
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
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)


class NodeResponse(BaseModel):
    node_name: str = Field(..., description="The node name of the response")
    docstring: str = Field(..., description="The docstring of the response")
    code: str = Field(..., description="The code of the response")


class RAGResponse(BaseModel):
    citations: List[str] = Field(
        ..., description="List of file names referenced in the response"
    )
    response: List[NodeResponse]


class DebugRAGAgent:
    def __init__(self, sql_db, llm, mini_llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iter = os.getenv("MAX_ITER", 5)
        self.sql_db = sql_db
        self._llm = None  # Initialize as None
        self._llm_provider = ProviderService(sql_db, user_id)
        self.get_code_from_node_id = get_code_from_node_id_tool(sql_db, user_id)
        self.get_code_from_multiple_node_ids = get_code_from_multiple_node_ids_tool(
            sql_db, user_id
        )
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.get_nodes_from_tags = get_nodes_from_tags_tool(sql_db, user_id)
        self.ask_knowledge_graph_queries = get_ask_knowledge_graph_queries_tool(
            sql_db, user_id
        )
        self.get_node_neighbours_from_node_id = get_node_neighbours_from_node_id_tool(
            sql_db
        )
        self.get_code_file_structure = get_code_file_structure_tool(sql_db)

    async def _get_llm(self):
        if self._llm is None:
            self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.CREWAI)
        return self._llm

    async def create_agents(self):
        llm = await self._get_llm()  # Use the helper method
        query_agent = Agent(
            role="Context curation agent",
            goal=(
                "Handle querying the knowledge graph and refining the results to provide accurate and contextually rich responses."
            ),
            backstory=f"""
                You are a highly efficient and intelligent RAG agent capable of querying complex knowledge graphs and refining the results to generate precise and comprehensive responses.
                Your tasks include:
                1. Analyzing the user's query and formulating an effective strategy to extract relevant information from the code knowledge graph.
                2. Executing the query with minimal iterations, ensuring accuracy and relevance.
                3. Refining and enriching the initial results to provide a detailed and contextually appropriate response.
                4. Maintaining traceability by including relevant citations and references in your output.
                5. Including relevant citations in the response.

                You must adhere to the specified {self.max_iter} iterations to optimize performance and reduce latency.
            """,
            tools=[
                self.get_nodes_from_tags,
                self.ask_knowledge_graph_queries,
                self.get_code_from_multiple_node_ids,
                self.get_code_from_probable_node_name,
                self.get_node_neighbours_from_node_id,
                self.get_code_file_structure,
            ],
            allow_delegation=False,
            verbose=True,
            llm=llm,
            max_iter=self.max_iter,
        )

        return query_agent

    async def create_tasks(
        self,
        query: str,
        project_id: str,
        chat_history: List,
        node_ids: List[NodeContext],
        file_structure: str,
        code_results: List[Dict[str, Any]],
        query_agent,
    ):
        if not node_ids:
            node_ids = []

        combined_task = Task(
            description=f"""
            Adhere to {self.max_iter} iterations max. Analyze input:

            - Chat History: {chat_history}
            - Query: {query}
            - Project ID: {project_id}
            - User Node IDs: {[node.model_dump() for node in node_ids]}
            - File Structure upto depth 4:
{file_structure}
            - Code Results for user node ids: {code_results}


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

            -   Always traverse directories before attempting to access files
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
            """,
            expected_output=(
                "Markdown formatted chat response to user's query grounded in provided code context and tool results"
            ),
            agent=query_agent,
        )

        return combined_task

    async def run(
        self,
        query: str,
        project_id: str,
        chat_history: List,
        node_ids: List[NodeContext],
        file_structure: str,
    ) -> AsyncGenerator[str, None]:
        agentops.init(
            os.getenv("AGENTOPS_API_KEY"), default_tags=["openai-gpt-notebook"]
        )
        code_results = []
        if len(node_ids) > 0:
            code_results = await GetCodeFromMultipleNodeIdsTool(
                self.sql_db, self.user_id
            ).run_multiple(project_id, [node.node_id for node in node_ids])
        debug_agent = await self.create_agents()
        debug_task = await self.create_tasks(
            query,
            project_id,
            chat_history,
            node_ids,
            file_structure,
            code_results,
            debug_agent,
        )

        read_fd, write_fd = os.pipe()

        async def kickoff():
            with os.fdopen(write_fd, "w", buffering=1) as write_file:
                with redirect_stdout(write_file):
                    crew = Crew(
                        agents=[debug_agent],
                        tasks=[debug_task],
                        process=Process.sequential,
                        verbose=True,
                    )
                    await crew.kickoff_async()

        agentops.end_session("Success")

        asyncio.create_task(kickoff())

        # Stream the output
        final_answer_streaming = False
        async with aiofiles.open(read_fd, mode="r") as read_file:
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


async def kickoff_debug_rag_agent(
    query: str,
    project_id: str,
    chat_history: List,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> AsyncGenerator[str, None]:
    provider_service = ProviderService(sql_db, user_id)
    crew_ai_llm = await provider_service.get_large_llm(agent_type=AgentType.CREWAI)
    crew_ai_mini_llm = await provider_service.get_small_llm(agent_type=AgentType.CREWAI)
    debug_agent = DebugRAGAgent(sql_db, crew_ai_llm, crew_ai_mini_llm, user_id)
    file_structure = await CodeProviderService(sql_db).get_project_structure_async(
        project_id
    )
    async for chunk in debug_agent.run(
        query, project_id, chat_history, node_ids, file_structure
    ):
        yield chunk
