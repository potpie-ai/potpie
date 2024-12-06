import os
from typing import Any, Dict, List

import agentops
from crewai import Agent, Crew, Process, Task

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.llm_provider.llm_provider_service import (
    LLMProviderService,
)
from app.modules.intelligence.prompts_provider.agent_prompts_provider import (
    AgentPromptsProvider,
)
from app.modules.intelligence.prompts_provider.agent_types import AgentLLMType
from app.modules.intelligence.tools.code_query_tools.get_code_file_structure import (
    get_code_file_structure_tool,
)
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
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)


class CodeGenerationAgent:
    def __init__(self, sql_db, llm, mini_llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iter = os.getenv("MAX_ITER", 15)
        self.sql_db = sql_db
        self.get_code_from_multiple_node_ids = get_code_from_multiple_node_ids_tool(
            sql_db, user_id
        )
        self.get_node_neighbours = get_node_neighbours_from_node_id_tool(sql_db)
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(
            sql_db, user_id
        )
        self.query_knowledge_graph = get_ask_knowledge_graph_queries_tool(
            sql_db, user_id
        )
        self.get_nodes_from_tags = get_nodes_from_tags_tool(sql_db, user_id)
        self.get_file_structure = get_code_file_structure_tool(sql_db)
        self.llm = llm
        self.mini_llm = mini_llm
        self.user_id = user_id

    async def create_agents(self):
        agent_prompt = await AgentPromptsProvider.get_agent_prompt(
            agent_id="code_generator", user_id=self.user_id, db=self.sql_db
        )
        code_generator = Agent(
            role=agent_prompt["role"],
            goal=agent_prompt["goal"],
            backstory=agent_prompt["backstory"],
            tools=[
                self.get_code_from_multiple_node_ids,
                self.get_node_neighbours,
                self.get_code_from_probable_node_name,
                self.query_knowledge_graph,
                self.get_nodes_from_tags,
                self.get_file_structure,
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iter,
        )

        return code_generator

    async def create_tasks(
        self,
        query: str,
        project_id: str,
        history: str,
        node_ids: List[NodeContext],
        code_results: List[Dict[str, Any]],
        code_generator,
    ):
        node_ids_list = [node.model_dump() for node in node_ids]
        task_prompt = await AgentPromptsProvider.get_task_prompt(
            task_id="code_generation_task",
            user_id=self.user_id,
            db=self.sql_db,
            query=query,
            project_id=project_id,
            history=history,
            node_ids=node_ids_list,
            code_results=code_results,
            max_iter=self.max_iter,
        )

        code_generation_task = Task(
            description=task_prompt,
            expected_output="User-friendly, clearly structured code changes with comprehensive dependency analysis, implementation details for ALL impacted files, and complete verification steps",
            agent=code_generator,
        )

        return code_generation_task

    async def run(
        self,
        query: str,
        project_id: str,
        history: str,
        node_ids: List[NodeContext],
    ) -> str:
        code_results = []
        if len(node_ids) > 0:
            code_results = await GetCodeFromMultipleNodeIdsTool(
                self.sql_db, self.user_id
            ).run_multiple(project_id, [node.node_id for node in node_ids])

        code_generator = await self.create_agents()
        generation_task = await self.create_tasks(
            query,
            project_id,
            history,
            node_ids,
            code_results,
            code_generator,
        )

        crew = Crew(
            agents=[code_generator],
            tasks=[generation_task],
            process=Process.sequential,
            verbose=False,
        )
        agentops.init(os.getenv("AGENTOPS_API_KEY"))
        result = await crew.kickoff_async()
        agentops.end_session("Success")
        return result


async def kickoff_code_generation_crew(
    query: str,
    project_id: str,
    history: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> str:
    provider_service = LLMProviderService(sql_db, user_id)
    crew_ai_mini_llm = provider_service.get_small_llm(agent_type=AgentLLMType.CREWAI)
    crew_ai_llm = provider_service.get_large_llm(agent_type=AgentLLMType.CREWAI)
    code_gen_agent = CodeGenerationAgent(sql_db, crew_ai_llm, crew_ai_mini_llm, user_id)
    result = await code_gen_agent.run(query, project_id, history, node_ids)
    return result
