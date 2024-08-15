import asyncio
from typing import AsyncGenerator, Optional
from uuid6 import uuid7
from datetime import datetime, timezone
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_model import Conversation, ConversationStatus
from app.modules.conversations.message.message_model import Message, MessageType
from app.modules.conversations.conversation.conversation_schema import ConversationInfoResponse, ConversationResponse, CreateConversationRequest
from app.modules.conversations.message.message_schema import MessageRequest, MessageResponse
from app.modules.conversations.message.message_service import MessageService
from app.modules.intelligence.agents.langchain_agents import LangChainAgent
from app.modules.projects.projects_service import ProjectService


class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.project_service = ProjectService(db)
        self.message_service = MessageService(db)

    async def perform_duckduckgo_search(self, query: str) -> AsyncGenerator[str, None]:
        agent = LangChainAgent()
        async for chunk in agent.run(f"Search for {query}"):
            yield chunk

    async def create_message(self, conversation_id: str, content: str, message_type: MessageType, sender_id: Optional[str] = None) -> Message:
        message_id = str(uuid7())
        new_message = Message(
            id=message_id,
            conversation_id=conversation_id,
            content=content,
            type=message_type,
            created_at=datetime.now(timezone.utc),
            sender_id=sender_id,
        )
        self.db.add(new_message)
        await self.commit_and_refresh(new_message)
        return new_message

    async def create_conversation(self, conversation: CreateConversationRequest):
        project_name = await self.project_service.get_project_name(conversation.project_ids)

        conversation_id = str(uuid7())
        new_conversation = Conversation(
            id=conversation_id,
            user_id=conversation.user_id,
            title=project_name,
            status=ConversationStatus.ACTIVE,
            project_ids=conversation.project_ids,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        agents = self.db.query(Agent).filter(Agent.id.in_(conversation.agent_ids)).all()
        new_conversation.agents.extend(agents)

        self.db.add(new_conversation)
        await self.commit()

        await self.create_message(conversation_id, f"Project {project_name} has been parsed successfully.", MessageType.SYSTEM_GENERATED)

        search_result = []
        async for result in self.perform_duckduckgo_search(project_name):
            search_result.append(result)

        combined_search_result = "\n".join(search_result)
        await self.create_message(conversation_id, combined_search_result, MessageType.AI_GENERATED)

        return conversation_id, combined_search_result

    async def store_message(self, conversation_id: str, message: MessageRequest, message_type: MessageType, user_id: Optional[str] = None) -> Message:
        return await self.create_message(conversation_id, message.content, message_type, user_id)

    async def message_stream(self, conversation_id: str, query: str) -> AsyncGenerator[str, None]:
        try:
            full_content = ""  # Initialize a variable to accumulate the content

            # Stream the response from DuckDuckGo search and accumulate the content
            async for content_update in self.perform_duckduckgo_search(query):
                full_content += content_update  # Accumulate the content
                yield f"data: {{'content': {content_update}}}\n\n"

            # Once the streaming is complete, insert the accumulated content as a single message
            await self.create_message(conversation_id, full_content.strip(), MessageType.AI_GENERATED)

        except Exception as e:
            self.db.rollback()  # Rollback the transaction in case of error
            raise e

    async def regenerate_last_message(self, conversation_id: str) -> AsyncGenerator[str, None]:
        try:
            # Step 1: Find the last human message
            last_human_message = (
                self.db.query(Message)
                .filter_by(conversation_id=conversation_id, type=MessageType.HUMAN)
                .order_by(Message.created_at.desc())
                .first()
            )

            if not last_human_message:
                raise ValueError("No human message found to regenerate from")

            # Step 2: Fetch all messages after the last human message by ID and delete them
            messages_to_delete = (
                self.db.query(Message)
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.id > last_human_message.id
                )
            )
            messages_to_delete.delete(synchronize_session='fetch')
            await self.commit()

            # Step 3: Stream the regenerated content and accumulate it
            full_content = ""
            async for chunk in self.perform_duckduckgo_search(last_human_message.content):
                full_content += chunk  # Accumulate the content for the new message
                yield f"data: {{'content': {chunk}}}\n\n"

            # Insert the new AI-generated message into the database
            await self.create_message(conversation_id, full_content.strip(), MessageType.AI_GENERATED)

        except Exception as e:
            self.db.rollback()  # Rollback the transaction in case of error
            raise e

    async def get_conversation(self, conversation_id: str) -> ConversationResponse:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise ValueError("Conversation not found")

        return ConversationResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            status=conversation.status.value,
            project_ids=conversation.project_ids,
            agent_ids=[agent.id for agent in conversation.agents],
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=[MessageResponse(
                id=message.id,
                conversation_id=message.conversation_id,
                content=message.content,
                sender_id=message.sender_id,
                type=message.type,
                created_at=message.created_at
            ) for message in conversation.messages]
        )

    async def get_conversation_info(self, conversation_id: str) -> ConversationInfoResponse:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise ValueError("Conversation not found")

        return ConversationInfoResponse(
            id=conversation.id,
            agent_ids=[agent.id for agent in conversation.agents],
            project_ids=conversation.project_ids,
            total_messages=len(conversation.messages)
        )

    async def get_conversation_messages(self, conversation_id: str, start: int, limit: int) -> list[MessageResponse]:
        messages = (
            self.db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .offset(start)
            .limit(limit)
            .all()
        )
        if not messages:
            return []

        return [
            MessageResponse(
                id=message.id,
                conversation_id=message.conversation_id,
                content=message.content,
                sender_id=message.sender_id,
                type=message.type,
                created_at=message.created_at
            ) for message in messages
        ]

    async def delete_conversation(self, conversation_id: str) -> dict:
        conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            raise ValueError("Conversation not found")

        self.db.delete(conversation)
        await self.commit()
        return {"status": "success"}

    async def commit_and_refresh(self, instance):
        """Commits the current transaction and refreshes the instance."""
        await asyncio.sleep(0)  # Yield control to the event loop
        self.db.commit()
        self.db.refresh(instance)

    async def commit(self):
        """Commits the current transaction."""
        await asyncio.sleep(0)  # Yield control to the event loop
        self.db.commit()

    async def stop_generation(self, conversation_id: str):
        pass
