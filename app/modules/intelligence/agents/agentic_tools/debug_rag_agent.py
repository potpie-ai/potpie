import os
from typing import Any, Dict, List

import agentops
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.github.github_service import GithubService
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


class DebugAgent:
    def __init__(self, sql_db, llm, mini_llm, user_id):
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
        self.get_node_neighbours_from_node_id = get_node_neighbours_from_node_id_tool(
            sql_db
        )
        self.llm = llm
        self.mini_llm = mini_llm
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
                self.get_code_from_multiple_node_ids,
                self.get_code_from_probable_node_name,
                self.get_node_neighbours_from_node_id,
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
            - File Structure: {file_structure}
            - Code Results for user node ids: {code_results}

            1. Analyze project structure:
               - Identify key directories, files, and modules
               - Guide search strategy and provide context
               - Locate files relevant to query
               - Use relevant file names with "Get Code and docstring From Probable Node Name" tool

            2. Initial context retrieval:
               - Analyze provided Code Results for user node ids
               - If code results are not relevant move to next step`

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
               - Format the file paths as follows (only include relevant project details from file path):
                 path: potpie/projects/username-reponame-branchname-userid/gymhero/models/training_plan.py
                 output: gymhero/models/training_plan.py

            Objective: Provide a comprehensive response with deep context and relevant file paths as citations.

            Note:
            - Prioritize "Get Code and docstring From Probable Node Name" tool for stacktraces or specific file/function mentions
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
    ) -> str:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        agentops.init(
            os.getenv("AGENTOPS_API_KEY"), default_tags=["openai-gpt-notebook"]
        )
        code_results = []
        if len(node_ids) > 0:
            code_results = await GetCodeFromMultipleNodeIdsTool(
                self.sql_db, self.user_id
            ).run_multiple(project_id, [node.node_id for node in node_ids])
        query_agent = await self.create_agents()
        query_task = await self.create_tasks(
            query,
            project_id,
            chat_history,
            node_ids,
            file_structure,
            code_results,
            query_agent,
        )

        crew = Crew(
            agents=[query_agent],
            tasks=[query_task],
            process=Process.sequential,
            verbose=False,
        )

        result = await crew.kickoff_async()
        agentops.end_session("Success")
        return result


async def kickoff_debug_crew(
    query: str,
    project_id: str,
    chat_history: List,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> str:
    debug_agent = DebugAgent(sql_db, llm, mini_llm, user_id)
    file_structure = GithubService(sql_db).get_project_structure(project_id)
    result = await debug_agent.run(
        query, project_id, chat_history, node_ids, file_structure
    )
    return result
