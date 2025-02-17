import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.base_agent_service import BaseAgentService
from app.modules.intelligence.agents.custom_agents.custom_agent_model import (
    CustomAgent as CustomAgentModel,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    Agent,
    AgentCreate,
    AgentUpdate,
    Task,
    TaskCreate,
)
from app.modules.intelligence.agents.custom_agents.runtime_agent import RuntimeAgent
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.key_management.secret_manager import SecretManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomAgentService(BaseAgentService):
    def __init__(self, db: Session):
        super().__init__(db)
        self.secret_manager = SecretManager()

    async def _get_agent_by_id_and_user(
        self, agent_id: str, user_id: str
    ) -> Optional[CustomAgentModel]:
        """Fetch a custom agent by ID and user ID"""
        try:
            agent = (
                self.db.query(CustomAgentModel)
                .filter(
                    CustomAgentModel.id == agent_id, CustomAgentModel.user_id == user_id
                )
                .first()
            )
            return agent
        except SQLAlchemyError as e:
            logger.error(f"Database error while fetching agent {agent_id}: {str(e)}")
            raise

    def _convert_to_agent_schema(self, custom_agent: CustomAgentModel) -> Agent:
        task_schemas = []
        for i, task in enumerate(custom_agent.tasks, start=1):
            expected_output = task.get("expected_output", {})
            if isinstance(expected_output, str):
                try:
                    expected_output = json.loads(expected_output)
                except json.JSONDecodeError:
                    raise ValueError(
                        "Invalid JSON format for expected_output in a task."
                    )
            task_schemas.append(
                Task(
                    id=i,
                    description=task["description"],
                    tools=task.get("tools", []),
                    expected_output=expected_output,
                )
            )

        return Agent(
            id=custom_agent.id,
            user_id=custom_agent.user_id,
            role=custom_agent.role,
            goal=custom_agent.goal,
            backstory=custom_agent.backstory,
            system_prompt=custom_agent.system_prompt,
            tasks=task_schemas,
            deployment_url=custom_agent.deployment_url,
            created_at=custom_agent.created_at,
            updated_at=custom_agent.updated_at,
            deployment_status=(
                custom_agent.deployment_status
                if custom_agent.deployment_status
                else "STOPPED"
            ),
        )

    async def create_agent(self, user_id: str, agent_data: AgentCreate) -> Agent:
        """Create a new custom agent with enhanced task descriptions"""
        try:
            # Extract tool IDs from tasks
            tool_ids = []
            for task in agent_data.tasks:
                tool_ids.extend(task.tools)

            # Validate tools
            available_tools = await self.fetch_available_tools(user_id)
            invalid_tools = [
                tool_id for tool_id in tool_ids if tool_id not in available_tools
            ]
            if invalid_tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"The following tool IDs are invalid: {', '.join(invalid_tools)}",
                )

            # Enhance task descriptions
            tasks_dict = [task.dict() for task in agent_data.tasks]
            enhanced_tasks = await self.enhance_task_descriptions(
                tasks_dict, agent_data.goal, available_tools, user_id
            )

            agent_id = str(uuid4())
            agent_model = CustomAgentModel(
                id=agent_id,
                user_id=user_id,
                role=agent_data.role,
                goal=agent_data.goal,
                backstory=agent_data.backstory,
                system_prompt=agent_data.system_prompt,
                tasks=enhanced_tasks,
            )

            self.db.add(agent_model)
            self.db.commit()
            self.db.refresh(agent_model)
            return self._convert_to_agent_schema(agent_model)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while creating agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create agent")
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def update_agent(
        self, agent_id: str, user_id: str, agent_data: AgentUpdate
    ) -> Optional[Agent]:
        """Update an existing custom agent"""
        try:
            agent = await self._get_agent_by_id_and_user(agent_id, user_id)
            if not agent:
                return None

            update_data = agent_data.dict(exclude_unset=True)
            if "tasks" in update_data:
                update_data["tasks"] = [task.dict() for task in agent_data.tasks]

            for key, value in update_data.items():
                setattr(agent, key, value)

            self.db.commit()
            self.db.refresh(agent)
            return self._convert_to_agent_schema(agent)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while updating agent {agent_id}: {str(e)}")
            raise

    async def delete_agent(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a custom agent"""
        try:
            agent = await self._get_agent_by_id_and_user(agent_id, user_id)
            if not agent:
                return {"success": False, "message": f"Agent {agent_id} not found"}

            self.db.delete(agent)
            self.db.commit()
            return {
                "success": True,
                "message": f"Agent {agent_id} successfully deleted",
            }
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while deleting agent {agent_id}: {str(e)}")
            raise

    async def get_agent(self, agent_id: str, user_id: str) -> Optional[Agent]:
        """Get a custom agent by ID"""
        agent = await self._get_agent_by_id_and_user(agent_id, user_id)
        return self._convert_to_agent_schema(agent) if agent else None

    def list_agents(self, user_id: str) -> List[Agent]:
        """List all custom agents for a user"""
        try:
            agents = (
                self.db.query(CustomAgentModel)
                .filter(CustomAgentModel.user_id == user_id)
                .all()
            )
            return [self._convert_to_agent_schema(agent) for agent in agents]
        except SQLAlchemyError as e:
            logger.error(f"Database error while listing agents: {str(e)}")
            raise

    async def execute_agent_runtime(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        node_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an agent at runtime without deployment"""
        agent_model = (
            self.db.query(CustomAgentModel)
            .filter(
                CustomAgentModel.id == agent_id, CustomAgentModel.user_id == user_id
            )
            .first()
        )

        if not agent_model:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent_config = {
            "user_id": agent_model.user_id,
            "role": agent_model.role,
            "goal": agent_model.goal,
            "backstory": agent_model.backstory,
            "system_prompt": agent_model.system_prompt,
            "tasks": agent_model.tasks,
        }
        llm = ProviderService(self.db, user_id).get_large_llm(AgentType.LANGCHAIN)
        runtime_agent = RuntimeAgent(llm, self.db, agent_config, self.secret_manager)
        try:
            result = await runtime_agent.run(
                agent_id, query, project_id, conversation_id, node_ids
            )
            return {"message": result["response"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def create_agent_plan_chain(self, user_id: str):
        """Create a LangChain for generating agent plans from prompts"""
        template = """You are an expert AI agent designer with advanced reasoning capabilities. Your task is to design a structured agent plan that uses the user's prompt and the available tools to achieve a clear, actionable goal.

User Prompt: {prompt}
Available Tools: {tools}

Using your expertise, create a comprehensive agent plan that includes:

1. **Role Definition**: Specify a clear professional title and role that aligns with the task.
2. **Goal Statement**: Derive a specific and actionable goal from the user prompt.
3. **Backstory**: Develop a concise professional backstory that establishes your credibility in AI agent design.
4. **System Prompt**: Write a detailed system prompt that guides the agent's behavior with precision.
5. **Task Breakdown**: Provide exactly **one** concrete task the agent must perform. The task should include:
   - A step-by-step description of the task.
   - The required tools to complete the task.
   - The expected output format as a JSON object.

Your final output must be a single, valid JSON object with the following structure:
{{
    "role": "Professional title and role",
    "goal": "Clear, actionable goal statement",
    "backstory": "Professional backstory",
    "system_prompt": "Detailed system prompt",
    "tasks": [
        {{
            "description": "Step-by-step task description",
            "tools": ["tool_id_1", "tool_id_2"],
            "expected_output": {{"key": "value"}}
        }}
    ]
}}

Ensure that your response is a properly formatted JSON object that can be parsed directly, with no extraneous text."""

        prompt = PromptTemplate(input_variables=["prompt", "tools"], template=template)

        llm = ProviderService(self.db, user_id).get_large_llm(AgentType.LANGCHAIN)
        return LLMChain(llm=llm, prompt=prompt)

    async def create_agent_from_prompt(
        self,
        prompt: str,
        user_id: str,
    ) -> Agent:
        """Create a custom agent from a natural language prompt"""
        # Create the planning chain
        chain = await self.create_agent_plan_chain(user_id)

        # Get available tools
        try:
            available_tools = ToolService(self.db, user_id).list_tools()
        except Exception as e:
            logger.error(f"Error fetching available tools: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to fetch available tools"
            )

        # Generate the agent plan
        try:
            result = await chain.ainvoke({"prompt": prompt, "tools": available_tools})

            # Extract the response text
            response_text = result.get("text", result)
            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Clean up the response text and parse JSON
            response_text = response_text.strip()
            try:
                # First try direct parsing
                plan_dict = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to find JSON content between curly braces
                start_idx = response_text.find("{")
                end_idx = response_text.rindex("}") + 1
                if start_idx != -1 and end_idx != -1:
                    json_content = response_text[start_idx:end_idx]
                    plan_dict = json.loads(json_content)
                else:
                    raise ValueError("Could not find valid JSON content in response")

            # Validate required fields
            required_fields = ["role", "goal", "backstory", "system_prompt", "tasks"]
            missing_fields = [
                field for field in required_fields if field not in plan_dict
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in response: {', '.join(missing_fields)}"
                )

            # Validate tasks structure
            if not isinstance(plan_dict["tasks"], list):
                raise ValueError("Tasks must be a list")

            for task in plan_dict["tasks"]:
                if not isinstance(task, dict):
                    raise ValueError("Each task must be an object")
                if "description" not in task:
                    raise ValueError("Each task must have a description")
                if "tools" not in task:
                    raise ValueError("Each task must have a tools list")
                if "expected_output" not in task:
                    task["expected_output"] = {"result": "string"}

            # Create the agent using the existing method
            agent_data = AgentCreate(
                role=plan_dict["role"],
                goal=plan_dict["goal"],
                backstory=plan_dict["backstory"],
                system_prompt=plan_dict["system_prompt"],
                tasks=[TaskCreate(**task) for task in plan_dict["tasks"]],
            )

            return await self.create_agent(user_id, agent_data)

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing error: {str(e)}\nResponse text: {response_text}"
            )
            raise ValueError(f"Failed to parse agent plan: {str(e)}")
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}\nResponse text: {response_text}")
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"Error creating agent from prompt: {str(e)}")
            raise ValueError("Failed to create agent from prompt")

    def create_task_enhancement_chain(self, user_id: str):
        template = """You are a task description enhancement expert. Your job is to transform a task description into a detailed execution plan.

        Original Task Description: {description}
        Task Goal: {goal}
        Available Tools: {tools}

        Analyze the task and create an enhanced description that:
        1. Shows understanding of the task's intent
        2. Provides step-by-step execution strategy
        3. Specifies when and how to use each tool
        4. Includes validation steps

        Follow patterns for each tool:

        get_nodes_from_tags:
        - Transform search queries into tags
        - Use for broad code search
        - Generate multiple tag variations

        get_code_file_structure:
        - Start with structure analysis
        - For hidden directories ("└── ..."):
        a. Get complete structure first
        b. Extract full file paths
        - Never skip structure retrieval

        get_code_from_probable_node_name:
        - Only use with complete paths
        - Never with directory paths
        - Validate retrieved content

        ask_knowledge_graph_queries:
        - Use function-based phrases
        - Include technical terms
        - Generate query variations

        get_node_neighbours_from_node_id:
        - Map dependencies
        - Check relationships
        - Follow code paths

        Response format:
        String with the following format:
        1. Analysis & Intent
        2. Step-by-Step Plan
        3. Tool Usage Guide
        """

        prompt = PromptTemplate(
            input_variables=["description", "goal", "tools"], template=template
        )

        llm = ProviderService(self.db, user_id).get_large_llm(AgentType.LANGCHAIN)

        return LLMChain(llm=llm, prompt=prompt)

    async def enhance_task_descriptions(
        self,
        tasks: List[Dict[str, Any]],
        goal: str,
        available_tools: List[str],
        user_id: str,
    ) -> List[Dict[str, Any]]:
        chain = self.create_task_enhancement_chain(user_id)
        enhanced_tasks = []

        for task in tasks:
            task_tools = [
                tool_id
                for tool_id in task.get("tools", [])
                if tool_id in available_tools
            ]

            enhanced_description = await chain.ainvoke(
                {"description": task["description"], "goal": goal, "tools": task_tools}
            )
            enhanced_task = task.copy()
            if "text" in enhanced_description:
                enhanced_task["description"] = enhanced_description["text"]
            else:
                enhanced_task["description"] = enhanced_description
            enhanced_tasks.append(enhanced_task)

        return enhanced_tasks

    async def fetch_available_tools(self, user_id: str) -> List[str]:
        """Fetches the list of available tool IDs."""
        try:
            tools = ToolService(self.db, user_id).list_tools()
            return [tool.id for tool in tools]
        except Exception as e:
            logger.error(f"Error fetching available tools: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to fetch available tools"
            )
