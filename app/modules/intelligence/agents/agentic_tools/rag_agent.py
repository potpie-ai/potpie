import os
from typing import List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

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
            You must adhere to the specified {self.max_iter} iterations to optimize performance and reduce latency.

            ## Input Details:
            - **Chat History:** {chat_history}
            - **Input Query:** {query}
            - **Project ID:** {project_id}
            - **User Provided Node IDs:** {[node.model_dump() for node in node_ids]}
            - **File Structure:** {file_structure}

            ## Step 0: Project Structure Analysis
            - Thoroughly analyze the provided file structure to understand the project's organization.
            - Identify key directories and their purposes based on common conventions and naming patterns.
            - Note any separation of concerns between different modules or components.
            - Identify important files like main entry points, configuration files, and core modules.
            - Use this structure to:
                1. Guide your search strategy in subsequent steps.
                2. Provide context for interpreting query results.
                3. Identify potential entry points or important modules related to the input query.
            - If the input query mentions specific functionalities, attempt to locate relevant files within the structure.
            - ALWAYS Use these relevant file names as probable node names for the "Get Code and docstring From Probable Node Name" tool to get context to answer users input query.
            - ALWAYS look at the file structure to fetch relevant file nodes to aswer the input query.
            ## Step 1: Initial Context Retrieval
            - If user provided node IDs are present:
            1. FIRST Use the "Get Code and docstring From Node ID" tool for each provided node ID.
            2. Analyze the retrieved docstrings and code.
            3. Determine if this information is sufficient to answer the input query.

            ## Step 2: Knowledge Graph Query (if needed)
            If the information from Step 1 is insufficient or if no node IDs were provided:
            1. Transform the original query for the knowledge graph tool:
            - Identify key concepts, code elements, and implied relationships.
            - Consider the context from chat history.
            - Determine the intent and key technical terms.
            - Transform into keyword phrases that might match docstrings:
                * Use concise, functionality-based phrases (e.g., "create document MongoDB collection").
                * Focus on verb-based keywords (e.g., "create", "define", "calculate").
                * Include docstring-related keywords like "parameters", "returns", "raises" when relevant.
                * Preserve key technical terms from the original query.
                * Generate multiple keyword variations to increase matching chances.
                * Be specific in keywords to improve match accuracy.
                * Phrase it as a docstring to improve match accuracy.
            2. Execute the transformed query using the knowledge graph tool.
            3. Analyze the returned response and determine if the returned nodes are sufficient to answer the input query accurately.

            ## Step 3: Additional Context Retrieval (if needed)
            If the knowledge graph results are insufficient:
            1. Extract probable node names (file, function names) from the input query or results.
            2. Use the "Get Code and docstring From Probable Node Name" tool for these extracted names.

            ## Step 3.5: FETCH USING TAGS AS A LAST RESORT ONLY IF REALLY REQUIRED
            - NEVER USE the "Get Nodes from Tags" tool to retrieve additional relevant nodes UNLESS REALLY REQUIRED.
            - USE THIS TOOL ONLY AFTER YOU HAVE TRIED TO GET RESPONSE USING ALL OTHER TOOLS AVAILABLE TO YOU.

            ## Step 4: Result Analysis and Enrichment
            - Evaluate the relevance of each result to the input query.
            - Identify potential gaps or redundancies in the information.
            - Develop a scoring mechanism considering:
            * Relevance to query
            * Code complexity
            * Hierarchical importance in the codebase
            * Frequency of references
            - For highly-ranked results, determine additional valuable context.
            - Retrieve code only if the docstring is insufficient to answer the input query.
            - Ensure retrieved code is complete and self-contained.



            ## Step 5: Response Composition
            - Organize the re-ranked and enriched results logically.
            - Include relevant citations and references to ensure traceability.
            - Provide a comprehensive and focused response that answers the user's query and offers a deeper understanding of the relevant code and its context within the project.
            - Maintain a balance between breadth and depth of information.

            ## Step 6: Final Review
            - Review the compiled results for overall coherence and relevance.
            - Identify any remaining gaps or potential improvements for future queries.

            ## Objective:
            Provide a comprehensive and focused response that not only answers the user's query but also offers a deeper understanding of the relevant code and its context within the project. Maintain a balance between breadth and depth of information retrieval.
            Include the file paths of relevant nodes as citations in the response.

            ## Note:
            - If a stacktrace or mention of a file/function is present in the original query, use the "Get Code and docstring From Probable Node Name" tool with the probable node names extracted from the stacktrace or mention for additional context before proceeding with other steps.
            - Always use available tools as directed in the original instructions.
            - If insufficient information is found at any stage, proceed to the next step in the algorithm.
            """,
            expected_output=(
                "Curated set of responses  that provide deep context to the user's query along with relevant file paths as citations."
                "Ensure that your output ALWAYS follows the structure outlined in the following Pydantic model:\n"
                f"{RAGResponse.model_json_schema()}"
            ),
            agent=query_agent,
            output_pydantic=RAGResponse,
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

        crew = Crew(
            agents=[query_agent],
            tasks=[query_task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            inputs={"user_id": self.user_id},
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
    user_id: str,
) -> str:
    rag_agent = RAGAgent(sql_db, llm, user_id)
    file_structure = GithubService(sql_db).get_project_structure(project_id)
    result = await rag_agent.run(
        query, project_id, chat_history, node_ids, file_structure
    )
    return result
