from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    Agent,
    AgentCreate,
    AgentUpdate,
    AgentVisibility,
)
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentService,
)
from app.modules.users.user_service import UserService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CustomAgentController:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        self.service = CustomAgentService(db)
        self.user_service = UserService(db)

    async def create_agent(self, user_id: str, agent_data: AgentCreate) -> Agent:
        """Create a new custom agent"""
        try:
            return await self.service.create_agent(user_id, agent_data)
        except Exception as e:
            logger.error(f"Error creating custom agent: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create custom agent: {str(e)}",
            )

    async def manage_agent_sharing(
        self,
        agent_id: str,
        owner_id: str,
        visibility: Optional[AgentVisibility] = None,
        shared_with_email: Optional[str] = None,
    ) -> Agent:
        """Manage agent sharing - set visibility or share with specific user"""
        try:
            # First verify the agent exists and belongs to the owner
            agent = await self.service.get_agent_model(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            if agent.user_id != owner_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to share this agent",
                )

            logger.info(
                f"Before update - Agent {agent_id} visibility: {agent.visibility}"
            )

            # Handle visibility changes first
            if visibility is not None:
                logger.info(f"Changing agent {agent_id} visibility to {visibility}")

                # If making private, remove all shares
                if visibility == AgentVisibility.PRIVATE:
                    logger.info(f"Making agent {agent_id} private")
                    updated_agent = await self.service.make_agent_private(
                        agent_id, owner_id
                    )
                    if not updated_agent:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to make agent private",
                        )
                    return updated_agent

                # Otherwise just update the visibility
                updated_agent = await self.service.update_agent(
                    agent_id,
                    owner_id,
                    AgentUpdate(visibility=visibility),
                )
                if not updated_agent:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to update agent visibility to {visibility}",
                    )

                # If we're not also sharing with a specific user, return now
                if not shared_with_email:
                    logger.info(
                        f"After update - Agent {agent_id} visibility: {updated_agent.visibility}"
                    )
                    return updated_agent

            # Handle sharing with specific user
            if shared_with_email:
                logger.info(f"Sharing agent {agent_id} with user {shared_with_email}")
                shared_user = await self.user_service.get_user_by_email(
                    shared_with_email
                )
                if not shared_user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"User with email {shared_with_email} not found",
                    )

                # Create share and update visibility if needed
                await self.service.create_share(agent_id, shared_user.uid)

                # Only update visibility if it's currently private and we haven't already changed it
                if agent.visibility == AgentVisibility.PRIVATE and visibility is None:
                    logger.info(f"Updating agent {agent_id} visibility to SHARED")
                    updated_agent = await self.service.update_agent(
                        agent_id,
                        owner_id,
                        AgentUpdate(visibility=AgentVisibility.SHARED),
                    )
                    if not updated_agent:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to update agent visibility to shared",
                        )
                    logger.info(
                        f"After update - Agent {agent_id} visibility: {updated_agent.visibility}"
                    )
                    return updated_agent

                # Get the current agent state
                current_agent = await self.service.get_agent(agent_id, owner_id)
                logger.info(
                    f"Current agent {agent_id} visibility: {current_agent.visibility}"
                )
                return current_agent

            # If we haven't returned by now, return the current agent state
            current_agent = await self.service.get_agent(agent_id, owner_id)
            logger.info(
                f"Current agent {agent_id} visibility: {current_agent.visibility}"
            )
            return current_agent

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in manage_agent_sharing: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to manage agent sharing: {str(e)}",
            )

    async def revoke_agent_access(
        self, agent_id: str, owner_id: str, user_email: str
    ) -> Agent:
        """Revoke a specific user's access to an agent"""
        try:
            # First verify the agent exists and belongs to the owner
            agent = await self.service.get_agent_model(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            if agent.user_id != owner_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to modify this agent's sharing",
                )

            # Get the user whose access is being revoked
            user_to_revoke = await self.user_service.get_user_by_email(user_email)
            if not user_to_revoke:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with email {user_email} not found",
                )

            logger.info(f"Revoking access to agent {agent_id} for user {user_email}")

            # Revoke the share
            success = await self.service.revoke_share(agent_id, user_to_revoke.uid)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No share found for agent {agent_id} with user {user_email}",
                )

            # Get the updated agent
            updated_agent = await self.service.get_agent(agent_id, owner_id)
            logger.info(
                f"Access revoked, agent {agent_id} visibility: {updated_agent.visibility}"
            )
            return updated_agent

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in revoke_agent_access: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to revoke agent access: {str(e)}",
            )

    async def list_agent_shares(self, agent_id: str, owner_id: str) -> List[str]:
        """List all emails this agent has been shared with"""
        try:
            # First verify the agent exists and belongs to the owner
            agent = await self.service.get_agent_model(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )

            # Check if user is the owner or has admin access
            if agent.user_id != owner_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to view this agent's shares",
                )

            logger.info(f"Listing all shares for agent {agent_id}")
            # Get the list of emails this agent is shared with
            emails = await self.service.list_agent_shares(agent_id)
            return emails

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in list_agent_shares: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list agent shares: {str(e)}",
            )

    async def list_agents(
        self,
        user_id: str,
        include_public: bool = True,
        include_shared: bool = True,
    ) -> List[Agent]:
        """List all agents accessible to the user"""
        try:
            return await self.service.list_agents(
                user_id, include_public, include_shared
            )
        except Exception as e:
            logger.error(f"Error listing agents for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list agents: {str(e)}",
            )

    async def update_agent(
        self, agent_id: str, user_id: str, agent_data: AgentUpdate
    ) -> Agent:
        """Update an existing custom agent"""
        try:
            agent = await self.service.update_agent(agent_id, user_id, agent_data)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            return agent
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating custom agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update custom agent: {str(e)}",
            )

    async def delete_agent(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a custom agent"""
        try:
            result = await self.service.delete_agent(agent_id, user_id)
            if not result["success"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=result["message"]
                )
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting custom agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete custom agent: {str(e)}",
            )

    async def get_agent(self, agent_id: str, user_id: str) -> Agent:
        """Get a custom agent by ID"""
        try:
            agent = await self.service.get_agent(agent_id, user_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            return agent
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching custom agent {agent_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch custom agent: {str(e)}",
            )

    async def create_agent_from_prompt(
        self,
        prompt: str,
        user_id: str,
    ) -> Agent:
        """Create a custom agent from a natural language prompt"""
        try:
            return await self.service.create_agent_from_prompt(
                prompt=prompt,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"Error creating agent from prompt: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create agent from prompt: {str(e)}",
            )
