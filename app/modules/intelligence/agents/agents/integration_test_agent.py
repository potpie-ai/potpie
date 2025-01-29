import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.llm_provider.llm_provider_service import (
    LLMProviderService,
)
from app.modules.intelligence.prompts_provider.agent_prompts_provider import (
    AgentPromptsProvider,
)
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.prompts.prompt_schema import PromptType
from app.modules.intelligence.prompts_provider.agent_types import AgentLLMType
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    get_code_from_multiple_node_ids_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)


class IntegrationTestAgent:
    def __init__(self, sql_db, llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.user_id = user_id
        self.sql_db = sql_db
        self.get_code_from_multiple_node_ids = get_code_from_multiple_node_ids_tool(
            sql_db, user_id
        )
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.llm = llm
        self.max_iterations = os.getenv("MAX_ITER", 15)
        self.prompt_service = PromptService(self.sql_db)


    async def create_agents(self):
        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        agent_prompt = await self.prompt_service.get_prompts(
            "integration_test_agent",
            [PromptType.SYSTEM],
            preferred_llm,
            max_iter=self.max_iterations,
        )
        
        integration_test_agent = Agent(
            role=agent_prompt["role"],
            goal=agent_prompt["goal"],
            backstory=agent_prompt["backstory"],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        return integration_test_agent

    class TestAgentResponse(BaseModel):
        response: str = Field(
            ...,
            description="String response containing the test plan and the test suite",
        )
        citations: List[str] = Field(
            ..., description="Exhaustive List of file names referenced in the response"
        )

    async def create_tasks(
        self,
        node_ids: List[NodeContext],
        project_id: str,
        query: str,
        graph: Dict[str, Any],
        history: List[str],
        integration_test_agent,
    ):
        node_ids = [node.node_id for node in node_ids]

        llm_provider_service = LLMProviderService.create(self.sql_db, self.user_id)
        preferred_llm, _ = await llm_provider_service.get_preferred_llm(self.user_id)
        task_prompt = await self.prompt_service.get_prompts(
            "integration_test_task",
            [PromptType.SYSTEM],
            preferred_llm,
            graph=graph,
            node_ids=node_ids,
            project_id=project_id,
            query=query,
            history=history,
            max_iterations=self.max_iterations,
            TestAgentResponse=self.TestAgentResponse.model_json_schema(),
        )
        

        integration_test_task = Task(
            description=task_prompt,
            expected_output=f"Write COMPLETE CODE for integration tests for each node based on the test plan. Ensure that your output ALWAYS follows the structure outlined in the following pydantic model:\n{self.TestAgentResponse.model_json_schema()}",
            agent=integration_test_agent,
            output_pydantic=self.TestAgentResponse,
            tools=[
                self.get_code_from_probable_node_name,
                self.get_code_from_multiple_node_ids,
            ],
            async_execution=True,
        )

        return integration_test_task

    async def run(
        self,
        project_id: str,
        node_ids: List[NodeContext],
        query: str,
        graph: Dict[str, Any],
        history: List,
    ) -> Dict[str, str]:
        integration_test_agent = await self.create_agents()
        integration_test_task = await self.create_tasks(
            node_ids,
            project_id,
            query,
            graph,
            history,
            integration_test_agent,
        )

        crew = Crew(
            agents=[integration_test_agent],
            tasks=[integration_test_task],
            process=Process.sequential,
            verbose=True,
        )

        result = await crew.kickoff_async()
        return result


async def kickoff_integration_test_agent(
    query: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    user_id,
    history: List[str],
) -> Dict[str, str]:
    if not node_ids:
        raise HTTPException(status_code=400, detail="No node IDs provided")
    graph = GetCodeGraphFromNodeIdTool(sql_db).run(project_id, node_ids[0].node_id)

    def extract_node_ids(node):
        node_ids = []
        for child in node.get("children", []):
            node_ids.extend(extract_node_ids(child))
        return node_ids

    def extract_unique_node_contexts(node, visited=None):
        if visited is None:
            visited = set()
        node_contexts = []
        if node["id"] not in visited:
            visited.add(node["id"])
            node_contexts.append(NodeContext(node_id=node["id"], name=node["name"]))
            for child in node.get("children", []):
                node_contexts.extend(extract_unique_node_contexts(child, visited))
        return node_contexts

    node_contexts = extract_unique_node_contexts(graph["graph"]["root_node"])
    provider_service = LLMProviderService(sql_db, user_id)
    crew_ai_llm = provider_service.get_large_llm(agent_type=AgentLLMType.CREWAI)
    integration_test_agent = IntegrationTestAgent(sql_db, crew_ai_llm, user_id)
    result = await integration_test_agent.run(
        project_id, node_contexts, query, graph, history
    )
    return result
