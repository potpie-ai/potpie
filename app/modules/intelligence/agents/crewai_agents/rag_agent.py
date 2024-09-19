import os
from typing import List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.kg_based_tools.code_tools import CodeTools
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_tools,
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
        self.kg_tools = CodeTools.get_kg_tools()
        self.code_tools = get_code_tools(self.sql_db)
        self.llm = llm

    async def create_agents(self):
        query_agent = Agent(
            role="Knowledge Graph Querier",
            goal="Query the knowledge graph based on the input query and retrieve the top k responses.",
            backstory="""You are an expert in querying complex knowledge graphs. Your task is to analyze the user's query and formulate the most effective strategy to extract relevant information from the code knowledge graph.
            Based on the conversation history and current query, you rephrase the knowledge graph query to be more specific and relevant to the codebase.
            """,
            tools=self.kg_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        rerank_agent = Agent(
            role="Re-ranker and Code Fetcher",
            goal="Optimize and enrich the query results for maximum relevance and context",
            backstory="You are a sophisticated algorithm personified, specializing in result optimization and context enrichment. Your task is to take the initial query results and transform them into the most informative and relevant response possible.",
            tools=self.code_tools,
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
            description=f"""Query the knowledge graph based on the input query and chat history and input node IDs.
              Chat History: {chat_history}
              Input Query: {query}
              Project ID: {project_id}
              User Provided Node IDs: {[node.model_dump() for node in node_ids]}
              If the knowledge graph query is not useful, you can use the 'Get Nodes from Tags' tool to get ALL the relevant nodes from the graph for that type. USE THIS ONLY WHEN NECESSARY AS THIS IS EXTREMELY LARGE CONTEXT.
              Step 1: Analyze the query
                - Identify key concepts, code elements, and implied relationships
                - Consider the context from chat history

              Step 2: Formulate knowledge graph query strategy
                - Analyze the original query to identify its intent and key technical terms.
                - Decide whether to use direct node lookup, similarity search, or graph traversal based on the query type.
                - Determine appropriate tags to filter results.
                - Transform the query to closely resemble a docstring structure for better similarity scores. Consider the following approaches:
                1. Rephrase as a statement describing functionality (e.g., "The XYZ class is used to...")
                2. Use verb-based descriptions for functions (e.g., "Creates..", "Defines...", "Calculates...", "Processes...", "Manages...")
                3. Incorporate docstring patterns like "Returns", "Raises", or "Args" when relevant.
                4. Preserve key technical terms from the original query.
                5. Generate multiple phrasings to increase matching chances.
                6. Be specific in your questions. Instead of asking "List the HTTP method associated with all API endpoints", ask "List the HTTP method associated with the get_data() function" This increases the chances of a match.

                Examples:
                1. Original: "How do I use the XYZ class?"
                Transformed:
                - "The XYZ class is used to..."
                - "Initializing and utilizing the XYZ class involves..."

                2. Original: "What are the parameters for the process_data function?"
                Transformed:
                - "The process_data function accepts the following parameters:"
                - "Args for process_data function include..."

                3. Original: "Explain the error handling in the API request module"
                Transformed:
                - "Error handling in the API request module is implemented by..."
                - "The API request module raises the following exceptions:"


              Step 3: Execute query
                - Use the available tools to query the knowledge graph
                - Analyze initial results

              Step 4: Refine and expand
                - Based on initial results, determine if additional queries are needed
                - Consider exploring related nodes or expanding the search scope

              Step 5: Reflect on results
                - Evaluate the relevance and completeness of the retrieved information
                - Identify any gaps or areas that need further exploration

              Throughout this process, maintain a balance between breadth and depth of information retrieval. Your goal is to provide a comprehensive yet focused set of results that directly address the user's query.
              If you are not able to find a lot of information, let the next task fetch the code from the nodes you already retrieved
            """,
            expected_output="A list of responses responses including the node_id, node name, docstring and similarity score that were retrieved from the knowledge graph",
            agent=query_agent,
        )

        rerank_task = Task(
            description=(
                f"""
                Step 1: Analyze initial results
                - Evaluate the relevance of each result to the original query
                - Identify potential gaps or redundancies in the information

                Step 2: Re-ranking strategy
                - Develop a scoring mechanism considering factors like:
                    * Relevance to query
                    * Code complexity
                    * Hierarchical importance in the codebase
                    * Frequency of references

                Step 3: Context enrichment
                - For each highly-ranked result, determine what additional context would be most valuable
                - Decide which related code snippets or documentation to fetch

                Step 4: Code retrieval
                - Use the available tools to fetch the necessary code and documentation
                - Retrieve the code only if the docstring is not enough to answer the query
                - Retrieve the code when it is not already included in the context
                - Ensure retrieved code is complete and self-contained

                Step 5: Result composition
                - Organize the re-ranked and enriched results in a logical structure
                - Ensure traceability by including relevant citations and references

                Step 6: Final reflection
                - Review the compiled results for overall coherence and relevance
                - Identify any remaining gaps or potential improvements for future queries

                Your goal is to provide a response that not only answers the user's query but also gives them a deeper understanding of the relevant code and its context within the project.
                "Chat History: {chat_history}
                "Query: {query}
                "Project ID: {project_id}
            """
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
