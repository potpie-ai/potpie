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


class BlastRadiusAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.tools_provider = tools_provider
        self.llm_provider = llm_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self):
        agent_config = AgentConfig(
            role="Blast Radius Analyzer",
            goal="Analyze the impact of code changes",
            backstory="You are an AI expert in analyzing how code changes affect the rest of the codebase.",
            tasks=[
                TaskConfig(
                    description=blast_radius_task_prompt,
                    expected_output="Comprehensive impact analysis of the code changes on the codebase and answers to the users query about them.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_nodes_from_tags",
                "ask_knowledge_graph_queries",
                "get_code_from_multiple_node_ids",
                "change_detection",
                "webpage_extractor",
                "web_search_tool",
                "github_tool",
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
                agent_type=AgentType.CODE_CHANGES,
            )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


blast_radius_task_prompt = """

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

    In the response, the patches contain the file patches for the changes.
    The changes contain the list of changes with the updated and entry point code. Entry point corresponds to the API/Consumer upstream of the function that the change was made in.
    The citations contain the list of file names referenced in the changed code and entry point code.

    You also have access the the query knowledge graph tool to answer natural language questions about the codebase during the analysis.
    Based on the response from the get code changes tool, formulate queries to ask details about specific changed code elements.
    1. Frame your query for the knowledge graph tool:
    - Identify key concepts, code elements, and implied relationships from the changed code.
    - Consider the context from the users query:
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

    Refer to the for any specific instructions and follow them.

    Based on the analysis, provide a structured inference of the blast radius:
    1. Summarize the direct changes
    2. List potential indirect effects
    3. Identify any critical areas that require careful testing
    4. Suggest any necessary refactoring or additional changes to mitigate risks
    6. If the changes are impacting multiple APIs/Consumers, then say so.
"""
