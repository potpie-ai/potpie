import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.custom_agents.custom_agent_model import (
    CustomAgent as CustomAgentModel,
    CustomAgentShare as CustomAgentShareModel,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    Agent,
    AgentCreate,
    AgentUpdate,
    Task,
    TaskCreate,
    AgentVisibility,
)
from app.modules.intelligence.agents.custom_agents.runtime_agent import (
    RuntimeAgent,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.key_management.secret_manager import SecretManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomAgentService:
    def __init__(self, db: Session):
        self.db = db
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
            if agent:
                logger.info(
                    f"Agent {agent_id} found for user {user_id}, visibility: {agent.visibility}"
                )
            else:
                logger.info(f"Agent {agent_id} not found for user {user_id}")
            return agent
        except SQLAlchemyError as e:
            logger.error(f"Database error while fetching agent {agent_id}: {str(e)}")
            raise

    async def get_agent_model(self, agent_id: str) -> Optional[CustomAgentModel]:
        """Fetch a custom agent model by ID without permission checks"""
        try:
            agent = (
                self.db.query(CustomAgentModel)
                .filter(CustomAgentModel.id == agent_id)
                .first()
            )
            if agent:
                logger.info(
                    f"Agent model {agent_id} found, visibility: {agent.visibility}"
                )
            else:
                logger.info(f"Agent model {agent_id} not found")
            return agent
        except SQLAlchemyError as e:
            logger.error(f"Database error while fetching agent {agent_id}: {str(e)}")
            raise

    async def create_share(
        self, agent_id: str, shared_with_user_id: str
    ) -> CustomAgentShareModel:
        """Create a share for an agent with another user"""
        try:

            # Get the agent to log its current state
            agent = await self.get_agent_model(agent_id)
            if agent:
                logger.info(f"Agent {agent_id} current visibility: {agent.visibility}")
            else:
                logger.warning(f"Agent {agent_id} not found when creating share")

            # Check if share already exists
            existing_share = (
                self.db.query(CustomAgentShareModel)
                .filter(
                    CustomAgentShareModel.agent_id == agent_id,
                    CustomAgentShareModel.shared_with_user_id == shared_with_user_id,
                )
                .first()
            )
            if existing_share:
                logger.info(
                    f"Share already exists for agent {agent_id} with user {shared_with_user_id}"
                )
                return existing_share

            # Create new share
            share = CustomAgentShareModel(
                id=str(uuid4()),
                agent_id=agent_id,
                shared_with_user_id=shared_with_user_id,
            )
            self.db.add(share)
            self.db.commit()

            # Get the agent again to verify its state after sharing
            agent = await self.get_agent_model(agent_id)
            if agent:
                logger.info(
                    f"Agent {agent_id} visibility after share creation: {agent.visibility}"
                )

            return share
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while creating share: {str(e)}")
            raise

    async def revoke_share(self, agent_id: str, shared_with_user_id: str) -> bool:
        """Revoke access to an agent for a specific user"""
        try:

            # Get the agent to log its current state
            agent = await self.get_agent_model(agent_id)
            if agent:
                logger.info(f"Agent {agent_id} current visibility: {agent.visibility}")
            else:
                logger.warning(f"Agent {agent_id} not found when revoking access")
                return False

            # Find the share to delete
            share = (
                self.db.query(CustomAgentShareModel)
                .filter(
                    CustomAgentShareModel.agent_id == agent_id,
                    CustomAgentShareModel.shared_with_user_id == shared_with_user_id,
                )
                .first()
            )

            if not share:
                logger.info(
                    f"No share found for agent {agent_id} with user {shared_with_user_id}"
                )
                return False

            self.db.delete(share)
            self.db.commit()
            logger.info(f"Share deleted successfully for agent {agent_id}")

            # Check if there are any remaining shares
            remaining_shares = (
                self.db.query(CustomAgentShareModel)
                .filter(CustomAgentShareModel.agent_id == agent_id)
                .count()
            )

            # If no more shares and visibility is SHARED, update to PRIVATE
            if remaining_shares == 0 and agent.visibility == AgentVisibility.SHARED:
                logger.info(
                    f"No more shares for agent {agent_id}, updating visibility to PRIVATE"
                )
                agent.visibility = AgentVisibility.PRIVATE.value
                self.db.commit()

            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while revoking share: {str(e)}")
            raise

    async def list_agent_shares(self, agent_id: str) -> list[str]:
        """List all emails this agent has been shared with"""
        try:

            # Get all user IDs this agent is shared with
            shares = (
                self.db.query(CustomAgentShareModel)
                .filter(CustomAgentShareModel.agent_id == agent_id)
                .all()
            )

            if not shares:
                logger.info(f"No shares found for agent {agent_id}")
                return []

            # Get all user emails from user IDs
            from app.modules.users.user_model import User

            shared_user_ids = [share.shared_with_user_id for share in shares]
            shared_users = (
                self.db.query(User).filter(User.uid.in_(shared_user_ids)).all()
            )

            emails = [user.email for user in shared_users] if shared_users else []
            logger.info(f"Agent {agent_id} is shared with {len(emails)} users")

            return emails
        except SQLAlchemyError as e:
            logger.error(f"Database error while listing agent shares: {str(e)}")
            raise

    async def make_agent_private(self, agent_id: str, user_id: str) -> Optional[Agent]:
        """Make an agent private, removing all shares and changing visibility"""
        try:

            # Get the agent and verify ownership
            agent = await self._get_agent_by_id_and_user(agent_id, user_id)
            if not agent:
                logger.warning(f"Agent {agent_id} not found for user {user_id}")
                return None

            # Delete all shares
            self.db.query(CustomAgentShareModel).filter(
                CustomAgentShareModel.agent_id == agent_id
            ).delete()

            # Update visibility to PRIVATE
            agent.visibility = AgentVisibility.PRIVATE.value
            self.db.commit()

            logger.info(f"Agent {agent_id} is now private")

            # Return the updated agent
            return self._convert_to_agent_schema(agent)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while making agent private: {str(e)}")
            raise

    async def list_agents(
        self, user_id: str, include_public: bool = False, include_shared: bool = True
    ) -> List[Agent]:
        """List all agents accessible to the user"""
        try:
            query = self.db.query(CustomAgentModel)

            # Base query for user's own agents
            filters = [CustomAgentModel.user_id == user_id]

            # Add public agents
            if include_public:
                filters.append(CustomAgentModel.visibility == AgentVisibility.PUBLIC)

            # Add shared agents
            if include_shared:
                shared_subquery = (
                    select(CustomAgentShareModel.agent_id)
                    .where(CustomAgentShareModel.shared_with_user_id == user_id)
                    .scalar_subquery()
                )
                filters.append(CustomAgentModel.id.in_(shared_subquery))

            # Combine all filters with OR
            query = query.filter(or_(*filters))

            agents = query.all()
            return [self._convert_to_agent_schema(agent) for agent in agents]
        except SQLAlchemyError as e:
            logger.error(f"Database error while listing agents: {str(e)}")
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

        # Convert visibility string to enum
        visibility = custom_agent.visibility
        try:
            # Try to convert string to enum
            visibility_enum = (
                AgentVisibility(visibility.lower())
                if visibility
                else AgentVisibility.PRIVATE
            )
        except ValueError:
            # If conversion fails, default to PRIVATE
            logger.warning(
                f"Invalid visibility value '{visibility}' for agent {custom_agent.id}, defaulting to PRIVATE"
            )
            visibility_enum = AgentVisibility.PRIVATE

        result = Agent(
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
            visibility=visibility_enum,
        )

        return result

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

            return self.persist_agent(user_id, agent_data, enhanced_tasks)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while creating agent: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create agent")
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def persist_agent(self, user_id, agent_data, tasks):
        agent_id = str(uuid4())
        if isinstance(tasks[0], TaskCreate):
            tasks = [task.dict() for task in tasks]

        agent_model = CustomAgentModel(
            id=agent_id,
            user_id=user_id,
            role=agent_data.role,
            goal=agent_data.goal,
            backstory=agent_data.backstory,
            system_prompt=agent_data.system_prompt,
            tasks=tasks,
        )

        self.db.add(agent_model)
        self.db.commit()
        self.db.refresh(agent_model)
        return self._convert_to_agent_schema(agent_model)

    async def update_agent(
        self, agent_id: str, user_id: str, agent_data: AgentUpdate
    ) -> Optional[Agent]:
        """Update an existing custom agent"""
        try:
            agent = await self._get_agent_by_id_and_user(agent_id, user_id)
            if not agent:
                logger.warning(f"Agent {agent_id} not found for user {user_id}")
                return None

            logger.info(
                f"Before update - Agent {agent_id} visibility: {agent.visibility}"
            )

            # Convert to dict and handle special fields
            update_data = agent_data.dict(exclude_unset=True)
            logger.info(f"Update data: {update_data}")

            # Handle tasks separately if present
            if "tasks" in update_data:
                update_data["tasks"] = [task.dict() for task in agent_data.tasks]

            # Handle visibility conversion from enum to string if present
            if "visibility" in update_data and update_data["visibility"] is not None:
                # Convert enum to string value
                logger.info(
                    f"Converting visibility from {update_data['visibility']} to {update_data['visibility'].value}"
                )
                update_data["visibility"] = update_data["visibility"].value

            # Apply all updates to the agent model
            for key, value in update_data.items():
                logger.info(f"Setting {key} = {value}")
                setattr(agent, key, value)

            # Explicitly commit changes to ensure they're saved
            logger.info(f"Committing changes to agent {agent_id}")
            self.db.commit()

            # Refresh the agent from the database to ensure we have the latest data
            self.db.refresh(agent)
            logger.info(
                f"After update - Agent {agent_id} visibility: {agent.visibility}"
            )

            # Convert to schema and return
            result = self._convert_to_agent_schema(agent)
            logger.info(f"Converted agent schema visibility: {result.visibility}")
            return result
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error while updating agent {agent_id}: {str(e)}")
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error while updating agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to update agent: {str(e)}"
            )

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

    async def get_agent(self, agent_id: str, user_id: str = None) -> Optional[Agent]:
        """Get a custom agent by ID with optional permission check"""
        agent_model = None
        logger.info(
            f"Getting agent {agent_id} for user {user_id if user_id else 'None'}"
        )

        # If user_id is provided, check permissions
        if user_id:
            # First check if user is the owner
            agent_model = await self._get_agent_by_id_and_user(agent_id, user_id)
            if agent_model:
                logger.info(f"User {user_id} is the owner of agent {agent_id}")

            # If not owner, check if agent is public or shared with user
            if not agent_model:
                agent_model = await self.get_agent_model(agent_id)
                if agent_model:
                    logger.info(
                        f"Agent {agent_id} found, visibility: {agent_model.visibility}"
                    )
                    # Check if agent is public
                    if agent_model.visibility == AgentVisibility.PUBLIC:
                        logger.info(
                            f"Agent {agent_id} is public, accessible to user {user_id}"
                        )
                    # Check if agent is shared with this user
                    elif agent_model.visibility == AgentVisibility.SHARED:
                        logger.info(
                            f"Agent {agent_id} is shared, checking if shared with user {user_id}"
                        )
                        share = (
                            self.db.query(CustomAgentShareModel)
                            .filter(
                                CustomAgentShareModel.agent_id == agent_id,
                                CustomAgentShareModel.shared_with_user_id == user_id,
                            )
                            .first()
                        )
                        if not share:
                            logger.info(
                                f"Agent {agent_id} is not shared with user {user_id}"
                            )
                            return None  # User doesn't have access to this shared agent
                        logger.info(f"Agent {agent_id} is shared with user {user_id}")
                    else:
                        logger.info(
                            f"Agent {agent_id} is private and user {user_id} is not the owner"
                        )
                        return None  # Private agent and user is not the owner
                else:
                    logger.info(f"Agent {agent_id} not found")
        else:
            # If no user_id provided, just get the agent without permission checks
            logger.info(
                f"No user_id provided, getting agent {agent_id} without permission checks"
            )
            agent_model = await self.get_agent_model(agent_id)
            if agent_model:
                logger.info(
                    f"Agent {agent_id} found without permission checks, visibility: {agent_model.visibility}"
                )
            else:
                logger.info(f"Agent {agent_id} not found")

        if agent_model:
            result = self._convert_to_agent_schema(agent_model)
            logger.info(
                f"Converted agent {agent_id} to schema, visibility: {result.visibility}"
            )
            return result
        else:
            return None

    async def execute_agent_runtime(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        node_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an agent at runtime without deployment"""
        logger.info(
            f"Executing agent {agent_id} for user {user_id} with query: {query}"
        )

        # First check if user is the owner
        agent_model = (
            self.db.query(CustomAgentModel)
            .filter(
                CustomAgentModel.id == agent_id, CustomAgentModel.user_id == user_id
            )
            .first()
        )

        # If not owner, check if agent is public or shared with user
        if not agent_model:
            logger.info(
                f"User {user_id} is not the owner of agent {agent_id}, checking visibility"
            )
            agent_model = (
                self.db.query(CustomAgentModel)
                .filter(CustomAgentModel.id == agent_id)
                .first()
            )
            if agent_model:
                logger.info(
                    f"Agent {agent_id} found, visibility: {agent_model.visibility}"
                )
                # Check if agent is public
                if agent_model.visibility == AgentVisibility.PUBLIC:
                    logger.info(
                        f"Agent {agent_id} is public, accessible to user {user_id}"
                    )
                # Check if agent is shared with this user
                elif agent_model.visibility == AgentVisibility.SHARED:
                    logger.info(
                        f"Agent {agent_id} is shared, checking if shared with user {user_id}"
                    )
                    share = (
                        self.db.query(CustomAgentShareModel)
                        .filter(
                            CustomAgentShareModel.agent_id == agent_id,
                            CustomAgentShareModel.shared_with_user_id == user_id,
                        )
                        .first()
                    )
                    if not share:
                        logger.info(
                            f"Agent {agent_id} is not shared with user {user_id}"
                        )
                        raise HTTPException(status_code=404, detail="Agent not found")
                    logger.info(f"Agent {agent_id} is shared with user {user_id}")
                else:
                    logger.info(
                        f"Agent {agent_id} is private and user {user_id} is not the owner"
                    )
                    raise HTTPException(status_code=404, detail="Agent not found")
            else:
                logger.info(f"Agent {agent_id} not found")
                raise HTTPException(status_code=404, detail="Agent not found")

        logger.info(f"Executing agent {agent_id} with role: {agent_model.role}")
        agent_config = {
            "user_id": agent_model.user_id,
            "role": agent_model.role,
            "goal": agent_model.goal,
            "backstory": agent_model.backstory,
            "system_prompt": agent_model.system_prompt,
            "tasks": agent_model.tasks,
        }
        runtime_agent = RuntimeAgent(self.db, agent_config)
        try:
            result = await runtime_agent.run(
                agent_id, query, project_id, project_name, conversation_id, node_ids
            )
            return {"message": result["response"]}

        except Exception as e:
            logger.error(f"Error executing agent {agent_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def create_agent_plan(
        self, user_id: str, prompt: str, tools: List[str]
    ) -> Dict[str, Any]:
        """Create a plan for the agent using LLM"""
        template = self.CREATE_AGENT_FROM_PROMPT

        formatted_prompt = template.format(prompt=prompt, tools=tools)
        messages = [{"role": "user", "content": formatted_prompt}]
        provider_service = ProviderService(self.db, user_id)
        response = await provider_service.call_llm(messages, config_type="chat")
        return response

    async def enhance_task_description(
        self, user_id: str, description: str, goal: str, tools: List[str]
    ) -> str:
        """Enhance a single task description using LLM"""
        template = self.TASK_ENHANCEMENT_PROMPT
        formatted_prompt = template.format(
            description=description, goal=goal, tools=tools
        )
        messages = [{"role": "user", "content": formatted_prompt}]
        provider_service = ProviderService(self.db, user_id)
        response = await provider_service.call_llm(messages, config_type="chat")
        return response

    async def create_agent_from_prompt(
        self,
        prompt: str,
        user_id: str,
    ) -> Agent:
        """Create a custom agent from a natural language prompt"""
        # Get available tools
        try:
            available_tools = ToolService(self.db, user_id).list_tools()
            tool_ids = [tool.id for tool in available_tools]
        except Exception as e:
            logger.error(f"Error fetching available tools: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to fetch available tools"
            )

        # Generate the agent plan
        try:
            response_text = await self.create_agent_plan(user_id, prompt, tool_ids)

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

            return self.persist_agent(user_id, agent_data, agent_data.tasks)

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

    async def enhance_task_descriptions(
        self,
        tasks: List[Dict[str, Any]],
        goal: str,
        available_tools: List[str],
        user_id: str,
    ) -> List[Dict[str, Any]]:
        enhanced_tasks = []

        for task in tasks:
            task_tools = [
                tool_id
                for tool_id in task.get("tools", [])
                if tool_id in available_tools
            ]

            enhanced_description = await self.enhance_task_description(
                user_id, task["description"], goal, task_tools
            )
            enhanced_task = task.copy()
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

    async def get_custom_agent(self, db: Session, user_id: str, agent_id: str):
        """Validate if an agent exists and belongs to the user or is shared with the user"""
        try:
            logger.info(f"Validating agent {agent_id} for user {user_id}")

            # First check if user is the owner
            agent = (
                db.query(CustomAgentModel)
                .filter(
                    CustomAgentModel.id == agent_id, CustomAgentModel.user_id == user_id
                )
                .first()
            )

            if agent:
                logger.info(f"User {user_id} is the owner of agent {agent_id}")
                return agent

            # If not owner, check if agent is public or shared with user
            agent = (
                db.query(CustomAgentModel)
                .filter(CustomAgentModel.id == agent_id)
                .first()
            )
            if not agent:
                logger.info(f"Agent {agent_id} not found")
                return None

            logger.info(f"Agent {agent_id} found, visibility: {agent.visibility}")

            # Check if agent is public
            if agent.visibility == AgentVisibility.PUBLIC:
                logger.info(f"Agent {agent_id} is public, accessible to user {user_id}")
                return agent
            # Check if agent is shared with this user
            elif agent.visibility == AgentVisibility.SHARED:
                logger.info(
                    f"Agent {agent_id} is shared, checking if shared with user {user_id}"
                )
                share = (
                    db.query(CustomAgentShareModel)
                    .filter(
                        CustomAgentShareModel.agent_id == agent_id,
                        CustomAgentShareModel.shared_with_user_id == user_id,
                    )
                    .first()
                )
                if not share:
                    logger.info(f"Agent {agent_id} is not shared with user {user_id}")
                    return None  # User doesn't have access to this shared agent
                logger.info(f"Agent {agent_id} is shared with user {user_id}")
                return agent
            else:
                logger.info(
                    f"Agent {agent_id} is private and user {user_id} is not the owner"
                )
                return None  # Private agent and user is not the owner
        except SQLAlchemyError as e:
            logger.error(f"Error validating agent {agent_id}: {str(e)}")
            raise e

    CREATE_AGENT_FROM_PROMPT = """
You are an expert AI agent designer specializing in creating optimal agent configurations for the Potpie system. Your goal is to analyze user input and create a comprehensive agent plan that either preserves detailed existing instructions or expands minimal prompts with best practices.

### INPUT ANALYSIS PHASE
User Prompt: {prompt}

Available Tools:
{tools}

### REASONING STEPS

## Step 1: Analyze the input prompt's detail level
- Is the prompt highly detailed with specific requirements?
- Or is it minimal/vague requiring expansion?
- What is the core objective the user is trying to accomplish?

## Step 2: Identify the key components needed
- What role would best accomplish this objective?
- What specific goal needs to be articulated?
- What professional backstory would establish credibility?
- What system prompt guidelines are needed?

## Step 3: Analyze available tools and their optimal applications
- For each available tool, identify:
  * What specific capabilities does it offer?
  * What types of tasks is it best suited for?
  * What limitations or constraints should be considered?
  * How can it be optimally combined with other tools?

## Step 4: Match requirements to appropriate tools
- Which tools are essential for the core requirements?
- Which additional tools could enhance the solution?
- What is the optimal sequence of tool usage?
- How should tools be configured for maximum effectiveness?

## Step 5: Design the comprehensive single task structure
- How can all required operations be consolidated into one robust task?
- What detailed step-by-step instructions will ensure complete success?
- What specific parameters should be passed to each tool?
- What validation steps should be included?
- What is the expected output format for the task?

### CRAFTING AN EFFECTIVE COMPREHENSIVE TASK DESCRIPTION

The task description is the MOST CRITICAL component of your agent plan. It directly determines how well the agent will reason and perform. Your single comprehensive task should:

1. Begin with a clear objective statement that sets the context
2. Include explicit reasoning prompts like "Let's think through this step by step" or "First, I need to understand..."
3. Break the complex operation into discrete, sequential steps within the single task
4. For each tool usage:
   - Explain WHY this specific tool is being used
   - Detail EXACTLY what parameters to use and their format
   - Describe HOW to interpret the results
   - Include validation checks for the output
5. Anticipate potential errors and include recovery strategies
6. Connect steps with logical transitions explaining the flow
7. End with verification steps to ensure the output meets requirements
8. Use specific examples where helpful

Example comprehensive task description structure:

[Objective Statement]

Let's think through this step by step:

1. First, I need to understand [specific aspect]. I'll use [Tool A] because [specific reason].
   - Input parameters: [exact parameter details]
   - Expected output: [specific output format]
   - Validation: [how to verify the output is correct]
   - If [potential error occurs], then [specific recovery action]

2. Next, I'll [next logical step] using [Tool B] to [specific purpose].
   - Input: [specific input derived from previous step]
   - Process: [detailed processing steps]
   - Output validation: [specific validation criteria]

3. **Tool-Specific Guidance**
   For each tool, provide detailed guidance on:

   **get_nodes_from_tags:**
   - Transform specific search requirements into effective tags
   - Generate multiple semantic variations to ensure comprehensive results
   - Format: Specify exactly how tags should be formatted
   - Validation: How to verify the search returned useful results
   - Handling: What to do if results are insufficient or too broad

   **get_code_file_structure:**
   - Begin with complete structure analysis before diving into details
   - For hidden directories ("└── ..."), explicitly get the complete structure
   - Extract and validate complete file paths
   - Never skip structure retrieval steps
   - Verification: How to confirm the structure is complete

   **get_code_from_probable_node_name:**
   - Use only with complete, validated file paths
   - Never use with directory paths
   - Verify the retrieved content is complete and relevant
   - Error handling: Steps to take if file cannot be found or is incomplete

   **ask_knowledge_graph_queries:**
   - Formulate precise queries using function-based phrasing
   - Include essential technical terminology
   - Generate multiple query variations for comprehensive results
   - Validation: How to assess if the returned information is sufficient

   **get_node_neighbours_from_node_id:**
   - Systematically map dependencies and relationships
   - Follow code paths methodically
   - Verify all connections are properly identified
   - Use for creating a complete understanding of code relationships


4. Finally, I'll [concluding action] to ensure [specific quality criteria].
   - Verification steps: [list of verification actions]
   - Output formatting: [format as human-readable markdown with appropriate headings, lists, and code blocks]


The complete result should be presented in clear, well-structured markdown that contains [specific elements] and satisfies [specific criteria].

### OUTPUT REQUIREMENTS

Based on your analysis, create a comprehensive agent plan as a valid JSON object with the following structure:

{{
    "role": "Precise professional title tailored to the specific task",
    "goal": "Clear, actionable goal statement preserving original intent and detail",
    "backstory": "Credible professional context that establishes expertise",
    "system_prompt": "Detailed guidelines that preserve original instructions while adding necessary context",
    "tasks": [
        {{
            "description": "Comprehensive, step-by-step instructions including specific tool usage guidance, parameters to use, and validation steps for the entire solution process",
            "tools": ["tool_id_1", "tool_id_2", "tool_id_3"],
            "expected_output": {{"markdown_result": "Human-readable markdown output with all necessary details and information and any relevant code blocks. Do not wrap the complete output in code blocks."}}
        }}
    ]
}}

IMPORTANT GUIDELINES:
1. If the user prompt is already detailed, PRESERVE all that detail in your plan - do not simplify or reduce specificity.
2. If the user prompt is minimal, EXPAND it with best practices based on the available tools.
3. STRICTLY LIMIT output to ONE TASK ONLY, but make this single task comprehensive enough to achieve the complete objective.
4. Provide DETAILED instructions on HOW to use each required tool, including specific parameters and validation steps.
5. The expected_output MUST specify that the final result will be in markdown format readable by humans, containing all necessary details and information.
6. ALWAYS incorporate explicit reasoning prompts in the task description to encourage step-by-step thinking.
7. Include error handling and recovery strategies in the task description.
8. Design the single task to incorporate iteration and refinement based on intermediate results if needed.
9. Ensure the task includes ALL necessary steps to achieve the complete objective.
10. The final output delivered to the user MUST be in human-readable markdown format with proper formatting, headings, lists, and code blocks if needed.

Return ONLY the valid JSON object with no additional text.
"""

    TASK_ENHANCEMENT_PROMPT = """
You are an advanced task planning specialist who transforms task descriptions into comprehensive execution plans that encourage optimal reasoning and tool usage. Your enhanced task descriptions will directly determine the reasoning quality and performance of AI agents using Potpie's tools.

### INPUT
Original Task Description: {description}
Task Goal: {goal}
Available Tools: {tools}

### REASONING STEPS

## Step 1: Deeply analyze the task requirements
- What is the core objective of this task?
- What specific challenges must be addressed?
- What expertise is needed to accomplish this task effectively?
- Is the original description already detailed (preserve detail) or minimal (expand strategically)?

## Step 2: Understand available tools and their optimal applications
For each available tool, analyze:
- What specific capabilities does it provide?
- When is this tool most effectively applied?
- What parameters and formatting requirements does it have?
- What limitations or edge cases should be considered?
- How can its output be validated?

## Step 3: Design an optimal execution strategy
- What is the logical sequence of operations needed?
- How do the available tools complement each other?
- What intermediate validations should be performed?
- What potential errors or edge cases might occur?
- How should the agent handle unexpected outcomes?

### CRAFTING THE ENHANCED TASK DESCRIPTION

Your enhanced task description should follow this structure:

1. **Task Objective Statement**
   A clear, concise statement of what needs to be accomplished that preserves the original intent.

2. **Reasoning Framework**
   Include explicit reasoning prompts like "Let's think through this step by step" or "I'll approach this methodically by..."

3. **Detailed Execution Plan**
   Break down the task into sequential steps, each containing:
   - Clear purpose of this step
   - Exact tool to use with specific parameters
   - Expected output and how to interpret it
   - Validation criteria to ensure correctness
   - Error handling and recovery strategies
   - Transition to the next step

4. **Tool-Specific Guidance**
   For each tool, provide detailed guidance on:

   **get_nodes_from_tags:**
   - Transform specific search requirements into effective tags
   - Generate multiple semantic variations to ensure comprehensive results
   - Format: Specify exactly how tags should be formatted
   - Validation: How to verify the search returned useful results
   - Handling: What to do if results are insufficient or too broad

   **get_code_file_structure:**
   - Begin with complete structure analysis before diving into details
   - For hidden directories ("└── ..."), explicitly get the complete structure
   - Extract and validate complete file paths
   - Never skip structure retrieval steps
   - Verification: How to confirm the structure is complete

   **get_code_from_probable_node_name:**
   - Use only with complete, validated file paths
   - Never use with directory paths
   - Verify the retrieved content is complete and relevant
   - Error handling: Steps to take if file cannot be found or is incomplete

   **ask_knowledge_graph_queries:**
   - Formulate precise queries using function-based phrasing
   - Include essential technical terminology
   - Generate multiple query variations for comprehensive results
   - Validation: How to assess if the returned information is sufficient

   **get_node_neighbours_from_node_id:**
   - Systematically map dependencies and relationships
   - Follow code paths methodically
   - Verify all connections are properly identified
   - Use for creating a complete understanding of code relationships

5. **Verification and Validation Strategy**
   - Specific criteria to verify the task was completed successfully
   - Tests to confirm all requirements were met
   - How to format and present the final output

6. **Error Recovery Approaches**
   - Anticipate potential failure points
   - Provide specific recovery strategies for each
   - Include alternative approaches when primary methods fail

The most effective task descriptions incorporate chain-of-thought reasoning by including phrases like:
- "First, I need to understand..."
- "Let me reason through this step by step..."
- "I should verify this by..."
- "If I encounter [specific error], I'll need to..."

### OUTPUT FORMAT

Your enhanced task description should be returned as a detailed, step-by-step string that maintains all original requirements while adding structure, reasoning guidance, and specific tool usage instructions. Do not simplify or reduce detail that was present in the original description.

Remember: The agent's performance depends directly on how thoroughly you structure this task description to encourage deliberate reasoning and methodical execution.
    """
