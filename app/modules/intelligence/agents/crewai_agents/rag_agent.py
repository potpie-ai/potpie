import os
from typing import List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.kg_based_tools.code_tools import CodeTools
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_tool,
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
    def __init__(self, sql_db, llm):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sql_db = sql_db
        self.code_tools = CodeTools.get_tools()
        self.get_code_tool = get_tool(self.sql_db)
        self.llm = llm

    async def create_agents(self):
        query_agent = Agent(
            role="Knowledge Graph Querier",
            goal="Query the knowledge graph based on the input query and retrieve the top k responses.",
            backstory="You specialize in querying knowledge graphs to find the most relevant vectors for a given query. Based on the conversation history and current query, you rephrase the knowledge graph query to be more specific and relevant to the codebase.",
            tools=[self.code_tools[0]],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        rerank_agent = Agent(
            role="Re-ranker",
            goal="Re-rank the top k responses and fetch the code for relevant node IDs, but skip the ones already queried.",
            backstory="You excel at analyzing and re-ranking responses based on query relevance, and retrieving code using corresponding node IDs.",
            tools=self.get_code_tool,
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return query_agent, rerank_agent

    async def create_tasks(
        self,
        query: str,
        project_id: str,
        chat_history: List,
        node_ids: List[NodeContext],
        query_agent,
        rerank_agent,
    ):
        if not node_ids:
            node_ids = []
        query_task = Task(
            description=f"Query the knowledge graph based on the input quer and chat history. Return the top k vector responses. Chat History: {chat_history}\n\nInput Query: {query}\n\nProject ID: {project_id}\n\n Node IDs: {[node.model_dump() for node in node_ids]}",
            expected_output="A list of vector similarity responses including the node_id, node name, docstring and similarity score",
            agent=query_agent,
        )

        rerank_task = Task(
            description=(
                "Re-rank the responses of the query task based on how relevant they are to the query. "
                "Discard the responses that are not relevant to the query. "
                "Based on the query, filter these responses and call the 'get code by ID' tool using the node_id from the responses FOR RELEVANT RESPONSES ONLY to fetch the corresponding code, "
                f"Chat History: {chat_history}"
                f"Query: {query}"
                f"Project ID: {project_id}"
            ),
            expected_output=f"The output of the task is curated context (file name, docstring, code) fetched from the knowledge graph along with required citations. Ensure that your output ALWAYS follows the structure outlined in the following pydantic model :\n{RAGResponse.model_json_schema()}",
            agent=rerank_agent,
            context=[query_task],
            output_pydantic=RAGResponse,
        )

        return query_task, rerank_task

    async def run(
        self,
        query: str,
        project_id: str,
        chat_history: List,
        node_ids: List[NodeContext],
    ) -> str:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        query_agent, rerank_agent = await self.create_agents()
        query_task, rerank_task = await self.create_tasks(
            query, project_id, chat_history, node_ids, query_agent, rerank_agent
        )

        crew = Crew(
            agents=[query_agent, rerank_agent],
            tasks=[query_task, rerank_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()
        return result


async def kickoff_rag_crew(
    query: str,
    project_id: str,
    chat_history: List,
    node_ids: List[NodeContext],
    sql_db,
    llm,
) -> str:
    rag_agent = RAGAgent(sql_db, llm)
    result = await rag_agent.run(query, project_id, chat_history, node_ids)
    return result
