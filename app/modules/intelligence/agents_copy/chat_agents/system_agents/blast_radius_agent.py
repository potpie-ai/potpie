from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_rag_agent import CrewAIRagAgent, AgentConfig, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse
from typing import List, Optional, AsyncGenerator


class BlastRadiusAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        project_id: str,
    ):
        self.rag_agent = CrewAIRagAgent(
            llm_provider,
            project_id,
            AgentConfig(
                role="Blast Radius Analyzer",
                goal="Analyze the impact of code changes",
                backstory="You are an AI expert in analyzing how code changes affect the rest of the codebase.",
                tasks=[
                    TaskConfig(
                        description=blast_radius_task_prompt,
                        expected_output="Comprehensive impact analysis of the code changes on the codebase and answers to the users query about them.",
                    )
                ],
            ),
            tools=[
                tools_provider.tools["get_nodes_from_tags"],
                tools_provider.tools["ask_knowledge_graph_queries"],
                tools_provider.tools["get_code_from_multiple_node_ids"],
                tools_provider.tools["webpage_extractor"],
                tools_provider.tools["github_tool"],
            ],
        )

    async def run(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> ChatAgentResponse:
        res = await self.run_stream(query, history, node_ids)
        async for response in res:
            return response

        # raise exception if we don't get a response
        raise Exception("response stream failed!!")

    async def run_stream(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        return self.rag_agent.run_stream(query, history, node_ids)


blast_radius_task_prompt = """
    Fetch the changes in the current branch for project {project_id} using the get code changes tool.
    The response of the fetch changes tool is in the following format:
    {ChangeDetectionResponse.model_json_schema()}
    In the response, the patches contain the file patches for the changes.
    The changes contain the list of changes with the updated and entry point code. Entry point corresponds to the API/Consumer upstream of the function that the change was made in.
    The citations contain the list of file names referenced in the changed code and entry point code.

    You also have access the the query knowledge graph tool to answer natural language questions about the codebase during the analysis.
    Based on the response from the get code changes tool, formulate queries to ask details about specific changed code elements.
    1. Frame your query for the knowledge graph tool:
    - Identify key concepts, code elements, and implied relationships from the changed code.
    - Consider the context from the users query: {query}.
    - Determine the intent and key technical terms.
    - Transform into keyword phrases that might match docstrings:
        * Use concise, functionality-based phrases (e.g., "creates document MongoDB collection").
        * Focus on verb-based keywords (e.g., "create", "define", "calculate").
        * Include docstring-related keywords like "parameters", "returns", "raises" when relevant.
        * Preserve key technical terms from the original query.
        * Generate multiple keyword variations to increase matching chances.
        * Be specific in keywords to improve match accuracy.
        * Ensure the query includes relevant details and follows a similar structure to enhance similarity search results.

    2. Execute your formulated query using the knowledge graph tool.

    Analyze the changes fetched and explain their impact on the codebase. Consider the following:
    1. Which functions or classes have been directly modified?
    2. What are the potential side effects of these changes?
    3. Are there any dependencies that might be affected?
    4. How might these changes impact the overall system behavior?
    5. Based on the entry point code, determine which APIs or consumers etc are impacted by the changes.

    Refer to the {query} for any specific instructions and follow them.

    Based on the analysis, provide a structured inference of the blast radius:
    1. Summarize the direct changes
    2. List potential indirect effects
    3. Identify any critical areas that require careful testing
    4. Suggest any necessary refactoring or additional changes to mitigate risks
    6. If the changes are impacting multiple APIs/Consumers, then say so.
"""
