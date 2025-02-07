import os
from typing import Dict, List

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.llm_provider.llm_provider_service import (
    LLMProviderService,
)
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.prompts.prompt_schema import PromptType
from app.modules.intelligence.prompts_provider.agent_types import AgentLLMType
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)


class UnitTestAgent:
    def __init__(self, sql_db, llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iterations = os.getenv("MAX_ITER", 15)
        self.sql_db = sql_db
        self.llm = llm
        self.user_id = user_id
        # Initialize tools with both sql_db and user_id
        self.get_code_from_node_id = get_code_from_node_id_tool(sql_db, user_id)
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.prompt_service = PromptService(self.sql_db)

    async def create_agents(self):
        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        agent_prompt = await self.prompt_service.get_prompts(
            "unit_test_agent",
            [PromptType.SYSTEM],
            preferred_llm,
            max_iter=self.max_iterations,
        )
        unit_test_agent = Agent(
            role=agent_prompt["role"],
            goal=agent_prompt["goal"],
            backstory=agent_prompt["backstory"],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iterations,
        )

        return unit_test_agent

    class TestAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response containing the Markdown formatted test plan and the test suite code block",
        )
        citations: List[str] = Field(
            ..., description="Exhaustive List of file names referenced in the response"
        )

    async def create_tasks(
        self,
        node_ids: List[NodeContext],
        project_id: str,
        query: str,
        history: List,
        unit_test_agent,
    ):
        node_ids_list = [node.node_id for node in node_ids]

        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        task_prompt = await self.prompt_service.get_prompts(
            "unit_test_task",
            [PromptType.SYSTEM],
            preferred_llm,
            node_ids_list=node_ids_list,
            project_id=project_id,
            query=query,
            history=history,
            max_iterations=self.max_iterations,
            TestAgentResponse=self.TestAgentResponse,
        )

        unit_test_task = Task(
            description=task_prompt,
            expected_output="Outline the test plan and write unit tests for each node based on the test plan.",
            agent=unit_test_agent,
            output_pydantic=self.TestAgentResponse,
            async_execution=True,
        )

        return unit_test_task

    async def run(
        self,
        project_id: str,
        node_ids: List[NodeContext],
        query: str,
        chat_history: List,
    ) -> Dict[str, str]:
        unit_test_agent = await self.create_agents()
        unit_test_task = await self.create_tasks(
            node_ids, project_id, query, chat_history, unit_test_agent
        )

        crew = Crew(
            agents=[unit_test_agent],
            tasks=[unit_test_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()

        return result


async def kickoff_unit_test_agent(
    query: str,
    chat_history: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    user_id,
) -> Dict[str, str]:
    if not node_ids:
        return {
            "error": "No function name is provided by the user. The agent cannot generate test plan or test code without specific class or function being selected by the user. Request the user to use the '@ followed by file or function name' feature to link individual functions to the message. "
        }
    provider_service = LLMProviderService(sql_db, user_id)
    crew_ai_llm = provider_service.get_large_llm(agent_type=AgentLLMType.CREWAI)
    unit_test_agent = UnitTestAgent(sql_db, crew_ai_llm, user_id)
    result = await unit_test_agent.run(project_id, node_ids, query, chat_history)
    return result
