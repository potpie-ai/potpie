from app.modules.intelligence.agents.chat_agents.adaptive_agent import AdaptiveAgent
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator
from app.modules.intelligence.prompts.classification_prompts import (
    AgentType,
)


class UnitTestAgent(ChatAgent):
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
            role="Test Plan and Unit Test Expert",
            goal="Create test plans and write unit tests based on user requirements",
            backstory="""You are a seasoned AI test engineer specializing in creating robust test plans and unit tests.
                        You aim to assist users effectively in generating and refining test plans and unit tests, ensuring they are comprehensive and tailored to the user's project requirements.""",
            tasks=[
                TaskConfig(
                    description=qna_task_prompt,
                    expected_output="Outline the test plan and write unit tests for each node based on the test plan.",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_code_from_node_id",
                "get_code_from_probable_node_name",
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
                agent_type=AgentType.UNIT_TEST,
            )

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code Graph context of the node_ids in query:\n {code_results}"
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


qna_task_prompt = """

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

    Process:

    **Analysis:**
    - Analyze the fetched code and docstrings to understand the functionality.
    - Identify the purpose, inputs, outputs, and potential side effects of each function/method.

    **Decision Making:**
    - Refer to the chat history to determine if a test plan or unit tests have already been generated.
    - If a test plan exists and the user requests modifications or additions, proceed accordingly without regenerating the entire plan.
    - If no existing test plan or unit tests are found, generate new ones based on the user's query.

    **Test Plan Generation:**
    Generate a test plan only if a test plan is not already present in the chat history or the user asks for it again.
    - For each function/method, create a detailed test plan covering:
        - Happy path scenarios
        - Edge cases (e.g., empty inputs, maximum values, type mismatches)
        - Error handling
        - Any relevant performance or security considerations
    - Format the test plan in two sections "Happy Path" and "Edge Cases" as neat bullet points

    **Unit Test Writing:**
    - Write complete unit tests based on the test plans.
    - Use appropriate testing frameworks and best practices.
    - Include clear, descriptive test names and explanatory comments.

    **Reflection and Iteration:**
    - Review the test plans and unit tests.
    - Ensure comprehensive coverage and correctness.
    - Make refinements as necessary, respecting the max iterations limit of max_iterations.

    **Response Construction:**
    - Provide the test plans and unit tests in your response.
    - Include any necessary explanations or notes.
    - Ensure the response is clear and well-organized.
"""
