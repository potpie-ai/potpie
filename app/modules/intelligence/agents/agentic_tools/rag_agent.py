import os
from typing import List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field
import agentops
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.github.github_service import GithubService
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
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


class RAGAgent:
    def __init__(self, sql_db, llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iter = os.getenv("MAX_ITER", 5)
        self.sql_db = sql_db
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
        self.llm = llm
        self.user_id = user_id

    async def create_agents(self):
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
                self.get_code_from_node_id,
                self.get_code_from_multiple_node_ids,
                self.get_code_from_probable_node_name,
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
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
            - File Structure: {file_structure}

            1. Analyze project structure:
               - Identify key directories, files, and modules
               - Guide search strategy and provide context
               - Locate files relevant to query
               - Use relevant file names with "Get Code and docstring From Probable Node Name" tool

            2. Initial context retrieval:
               - If node IDs provided, use "Get Code and docstring From Node ID" tool
               - Analyze retrieved data

            3. Knowledge graph query (if needed):
               - Transform query for knowledge graph tool
               - Execute query and analyze results

            4. Additional context retrieval (if needed):
               - Extract probable node names
               - Use "Get Code and docstring From Probable Node Name" tool

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

            Objective: Provide a comprehensive response with deep context and relevant file paths as citations.

            Note:
            - Prioritize "Get Code and docstring From Probable Node Name" tool for stacktraces or specific file/function mentions
            - Use available tools as directed
            - Proceed to next step if insufficient information found
            """,
            expected_output=(
                "Curated set of responses  that provide deep context to the user's query along with relevant file paths as citations."

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
    ) -> str:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        query_agent = await self.create_agents()
        query_task = await self.create_tasks(
            query, project_id, chat_history, node_ids, file_structure, query_agent
        )
        agentops.init(os.getenv("AGENTOPS_API_KEY"), default_tags=["openai-gpt-notebook"])


        crew = Crew(
            agents=[query_agent],
            tasks=[query_task],
            process=Process.sequential,
            verbose=False,
        )

        result = await crew.kickoff_async()
        agentops.end_session("Success")
        return result


async def kickoff_rag_crew(
    query: str,
    project_id: str,
    chat_history: List,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    user_id: str,
) -> str:
    rag_agent = RAGAgent(sql_db, llm, user_id)
    file_structure = GithubService(sql_db).get_project_structure(project_id)
    result = await rag_agent.run(
        query, project_id, chat_history, node_ids, file_structure
    )
    return result
