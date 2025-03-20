from typing import List

from sqlalchemy.orm import Session

from app.modules.users.user_schema import (
    UserConversationListResponse,
    UserProfileResponse,
)
from app.modules.users.user_service import UserService
from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent


class UserController:
    def __init__(self, db: Session):
        self.service = UserService(db)
        self.sql_db = db

    async def get_user_profile_pic(self, uid: str) -> UserProfileResponse:
        return await self.service.get_user_profile_pic(uid)

    async def get_conversations_for_user(
        self, user_id: str, start: int, limit: int
    ) -> List[UserConversationListResponse]:
        conversations = self.service.get_conversations_with_projects_for_user(
            user_id, start, limit
        )
        response = []
        for conversation in conversations:
            projects = conversation.projects
            repo_name = projects[0].repo_name
            branch_name = projects[0].branch_name

            agent_id = conversation.agent_ids[0] if conversation.agent_ids else None
            custom_agent_name = None

            custom_agent = self.sql_db.query(CustomAgent).filter_by(id=agent_id).first()
<<<<<<< HEAD
            display_agent_id = custom_agent.role if custom_agent else conversation.agent_ids[0]
            response.append(
                UserConversationListResponse(
                    id=conversation.id,
                    user_id=conversation.user_id,
                    title=conversation.title,
                    status=conversation.status,
                    project_ids=conversation.project_ids,
                    repository=repo_name,
                    branch=branch_name,
                    agent_id=display_agent_id,
                    created_at=conversation.created_at.isoformat(),
                    updated_at=conversation.updated_at.isoformat(),
                    shared_with_emails=conversation.shared_with_emails,
=======
            if custom_agent:
                custom_agent_name = custom_agent.role
                response.append(
                    UserConversationListResponse(
                        id=conversation.id,
                        user_id=conversation.user_id,
                        title=conversation.title,
                        status=conversation.status,
                        project_ids=conversation.project_ids,
                        repository=repo_name,
                        branch=branch_name,
                        agent_id=custom_agent_name,
                        created_at=conversation.created_at.isoformat(),
                        updated_at=conversation.updated_at.isoformat(),
                        shared_with_emails=conversation.shared_with_emails,
                    )
                )
            else:
                response.append(
                    UserConversationListResponse(
                        id=conversation.id,
                        user_id=conversation.user_id,
                        title=conversation.title,
                        status=conversation.status,
                        project_ids=conversation.project_ids,
                        repository=repo_name,
                        branch=branch_name,
                        agent_id=conversation.agent_ids[0],
                        created_at=conversation.created_at.isoformat(),
                        updated_at=conversation.updated_at.isoformat(),
                        shared_with_emails=conversation.shared_with_emails,
                    )
>>>>>>> d86445e4d4b269e97d5e1cebff026a917199b749
                )
            )

        return response
