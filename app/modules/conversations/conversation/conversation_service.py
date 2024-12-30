import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.types import Command, StreamWriter
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
    Visibility,
)
from app.modules.conversations.conversation.conversation_schema import (
    ConversationAccessType,
    ConversationInfoResponse,
    CreateConversationRequest,
)
from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
    MessageType,
)
from app.modules.conversations.message.message_schema import (
    MessageRequest,
    MessageResponse,
    NodeContext,
)
from app.modules.intelligence.agents.agent_factory import AgentFactory
from app.modules.intelligence.agents.agent_injector_service import AgentInjectorService
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentsService,
)
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.projects.projects_service import ProjectService
from app.modules.users.user_service import UserService
from app.modules.utils.posthog_helper import PostHogClient

logger = logging.getLogger(__name__)


class ConversationServiceError(Exception):
    pass


class ConversationNotFoundError(ConversationServiceError):
    pass


class MessageNotFoundError(ConversationServiceError):
    pass


class AccessTypeNotFoundError(ConversationServiceError):
    pass


class AccessTypeReadError(ConversationServiceError):
    pass


class SimplifiedAgentSupervisor:
    def __init__(self, db, provider_service):
        self.db = db
        self.provider_service = provider_service
        self.agents = {}
        self.classifier = None
        self.agents_service = AgentsService(db)
        self.agent_factory = AgentFactory(db, provider_service)

    async def initialize(self, user_id: str):
        available_agents = await self.agents_service.list_available_agents(
            current_user={"user_id": user_id}, list_system_agents=True
        )

        self.agents = {
            agent.id: self.agent_factory.get_agent(agent.id, user_id)
            for agent in available_agents
        }

        self.llm = await self.provider_service.get_small_llm(user_id)

        self.classifier_prompt = """
        Given the user query and the current agent ID, select the most appropriate agent by comparing the query’s requirements with each agent’s specialties.

        Query: {query}
        Current Agent ID: {agent_id}

        Available agents and their specialties:
        {agent_descriptions}

        Follow the instructions below to determine the best matching agent and provide a confidence score:

        Analysis Instructions (DO NOT include these instructions in the final answer):
        1. **Semantic Analysis:**
        - Identify the key topics, technical terms, and the user’s intent from the query.
        - Compare these elements to each agent’s detailed specialty description.
        - Focus on specific skills, tools, frameworks, and domain expertise mentioned.

        2. **Contextual Weighting:**
        - If the query strongly aligns with the current agent’s known capabilities, add +0.15 confidence for direct core expertise and +0.1 for related domain knowledge.
        - If the query introduces new topics outside the current agent’s domain, do not apply the current agent bias. Instead, evaluate all agents equally based on their described expertise.

        3. **Multi-Agent Evaluation:**
        - Consider all agents’ described specialties thoroughly, not just the current agent.
        - For overlapping capabilities, favor the agent with more specialized expertise or more relevant tools/methodologies.
        - If no agent clearly surpasses a 0.5 confidence threshold, select the agent with the highest confidence score, even if it is below 0.5.

        4. **Confidence Scoring Guidelines:**
        - 0.9-1.0: Ideal match with the agent’s core, primary expertise.
        - 0.7-0.9: Strong match with the agent’s known capabilities.
        - 0.5-0.7: Partial or related match, not a direct specialty.
        - Below 0.5: Weak match; consider if another agent is more suitable, but still choose the best available option.

        Final Output Requirements:
        - Return ONLY the chosen agent_id and the confidence score in the format:
        `agent_id|confidence`

        Examples:
        - Direct expertise match: `debugging_agent|0.95`
        - Related capability (current agent): `current_agent_id|0.75`
        - Need different expertise: `ml_training_agent|0.85`
        - Overlapping domains, choose more specialized: `choose_higher_expertise_agent|0.80`
        """

        self.agent_descriptions = "\n".join(
            [f"- {agent.id}: {agent.description}" for agent in available_agents]
        )

    class State(TypedDict):
        query: str
        project_id: str
        conversation_id: str
        response: Optional[str]
        agent_id: Optional[str]
        user_id: str
        node_ids: List[NodeContext]

    async def classifier_node(self, state: State) -> Command:
        """Classifies the query and routes to appropriate agent"""
        if not state.get("query"):
            return Command(update={"response": "No query provided"}, goto=END)

        # Classification using LLM with enhanced prompt
        prompt = self.classifier_prompt.format(
            query=state["query"],
            agent_id=state["agent_id"],
            agent_descriptions=self.agent_descriptions,
        )
        response = await self.llm.ainvoke(prompt)
        response = response.content.strip("`")
        try:
            agent_id, confidence = response.split("|")
            confidence = float(confidence)
        except (ValueError, TypeError):
            return Command(
                update={"response": "Error in classification format"}, goto=END
            )
        if confidence < 0.5 or agent_id not in self.agents:
            return Command(
                update={"agent_id": state["agent_id"]}, goto=state["agent_id"]
            )

        return Command(update={"agent_id": agent_id}, goto=agent_id)

    async def agent_node(self, state: State, writer: StreamWriter):
        """Creates a node function for a specific agent"""
        agent = self.agents[state["agent_id"]]
        async for chunk in agent.run(
            query=state["query"],
            project_id=state["project_id"],
            conversation_id=state["conversation_id"],
            user_id=state["user_id"],
            node_ids=state["node_ids"],
        ):
            if isinstance(chunk, str):
                writer(chunk)

    def build_graph(self) -> StateGraph:
        """Builds the graph with classifier and agent nodes"""
        builder = StateGraph(self.State)

        # Add classifier as entry point
        builder.add_node("classifier", self.classifier_node)

        # Add agent nodes
        for agent_id in self.agents:
            builder.add_node(agent_id, self.agent_node)
            builder.add_edge(agent_id, END)

        builder.set_entry_point("classifier")
        return builder.compile()

    async def process_query(
        self,
        query: str,
        project_id: str,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext],
        agent_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Main method to process queries"""
        state = {
            "query": query,
            "project_id": project_id,
            "conversation_id": conversation_id,
            "response": None,
            "user_id": user_id,
            "node_ids": node_ids,
            "agent_id": agent_id,
        }

        graph = self.build_graph()
        async for chunk in graph.astream(state, stream_mode="custom"):
            yield chunk


class ConversationService:
    def __init__(
        self,
        db: Session,
        user_id: str,
        user_email: str,
        project_service: ProjectService,
        history_manager: ChatHistoryService,
        provider_service: ProviderService,
        agent_injector_service: AgentInjectorService,
        custom_agent_service: CustomAgentsService,
    ):
        self.sql_db = db
        self.user_id = user_id
        self.user_email = user_email
        self.project_service = project_service
        self.history_manager = history_manager
        self.provider_service = provider_service
        self.agent_injector_service = agent_injector_service
        self.custom_agent_service = custom_agent_service

    @classmethod
    def create(cls, db: Session, user_id: str, user_email: str):
        project_service = ProjectService(db)
        history_manager = ChatHistoryService(db)
        provider_service = ProviderService(db, user_id)
        agent_injector_service = AgentInjectorService(db, provider_service, user_id)
        custom_agent_service = CustomAgentsService()
        return cls(
            db,
            user_id,
            user_email,
            project_service,
            history_manager,
            provider_service,
            agent_injector_service,
            custom_agent_service,
        )

    async def check_conversation_access(
        self, conversation_id: str, user_email: str
    ) -> str:
        user_service = UserService(self.sql_db)
        user_id = user_service.get_user_id_by_email(user_email)

        # Retrieve the conversation
        conversation = (
            self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
        )
        if not conversation:
            return (
                ConversationAccessType.NOT_FOUND
            )  # Return 'not found' if conversation doesn't exist

        if not conversation.visibility:
            conversation.visibility = Visibility.PRIVATE

        if conversation.visibility == Visibility.PUBLIC:
            return ConversationAccessType.READ

        if user_id == conversation.user_id:  # Check if the user is the creator
            return ConversationAccessType.WRITE  # Creator can write
        # Check if the conversation is shared
        if conversation.shared_with_emails:
            shared_user_ids = user_service.get_user_ids_by_emails(
                conversation.shared_with_emails
            )
            if shared_user_ids is None:
                return ConversationAccessType.NOT_FOUND
            # Check if the current user ID is in the shared user IDs
            if user_id in shared_user_ids:
                return ConversationAccessType.READ  # Shared user can only read
        return ConversationAccessType.NOT_FOUND

    async def create_conversation(
        self, conversation: CreateConversationRequest, user_id: str
    ) -> tuple[str, str]:
        try:
            if not self.agent_injector_service.validate_agent_id(
                user_id, conversation.agent_ids[0]
            ):
                raise ConversationServiceError(
                    f"Invalid agent_id: {conversation.agent_ids[0]}"
                )

            project_name = await self.project_service.get_project_name(
                conversation.project_ids
            )

            title = (
                conversation.title.strip().replace("Untitled", project_name)
                if conversation.title
                else project_name
            )

            conversation_id = self._create_conversation_record(
                conversation, title, user_id
            )

            asyncio.create_task(
                CodeProviderService(self.sql_db).get_project_structure_async(
                    conversation.project_ids[0]
                )
            )

            await self._add_system_message(conversation_id, project_name, user_id)

            return conversation_id, "Conversation created successfully."
        except IntegrityError as e:
            logger.error(f"IntegrityError in create_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to create conversation due to a database integrity error."
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in create_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "An unexpected error occurred while creating the conversation."
            ) from e

    def _create_conversation_record(
        self, conversation: CreateConversationRequest, title: str, user_id: str
    ) -> str:
        conversation_id = str(uuid7())
        new_conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title,
            status=ConversationStatus.ACTIVE,
            project_ids=conversation.project_ids,
            agent_ids=conversation.agent_ids,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.sql_db.add(new_conversation)
        self.sql_db.commit()
        logger.info(
            f"Project id : {conversation.project_ids[0]} Created new conversation with ID: {conversation_id}, title: {title}, user_id: {user_id}, agent_id: {conversation.agent_ids[0]}"
        )
        provider_name = self.provider_service.get_llm_provider_name()
        PostHogClient().send_event(
            user_id,
            "create Conversation Event",
            {
                "project_ids": conversation.project_ids,
                "agent_ids": conversation.agent_ids,
                "llm": provider_name,
            },
        )
        return conversation_id

    async def _add_system_message(
        self, conversation_id: str, project_name: str, user_id: str
    ):
        content = f"You can now ask questions about the {project_name} repository."
        try:
            self.history_manager.add_message_chunk(
                conversation_id, content, MessageType.SYSTEM_GENERATED, user_id
            )
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.SYSTEM_GENERATED, user_id
            )
            logger.info(
                f"Added system message to conversation {conversation_id} for user {user_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to add system message to conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to add system message to the conversation."
            ) from e

    async def store_message(
        self,
        conversation_id: str,
        message: MessageRequest,
        message_type: MessageType,
        user_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")
            self.history_manager.add_message_chunk(
                conversation_id, message.content, message_type, user_id
            )
            self.history_manager.flush_message_buffer(
                conversation_id, message_type, user_id
            )
            logger.info(f"Stored message in conversation {conversation_id}")
            provider_name = await self.provider_service.get_llm_provider_name()

            PostHogClient().send_event(
                user_id,
                "message post event",
                {"conversation_id": conversation_id, "llm": provider_name},
            )
            if message_type == MessageType.HUMAN:
                conversation = await self._get_conversation_with_message_count(
                    conversation_id
                )
                if not conversation:
                    raise ConversationNotFoundError(
                        f"Conversation with id {conversation_id} not found"
                    )

                # Check if this is the first human message
                if conversation.human_message_count == 1:
                    new_title = await self._generate_title(
                        conversation, message.content
                    )
                    await self._update_conversation_title(conversation_id, new_title)

                project_id = (
                    conversation.project_ids[0] if conversation.project_ids else None
                )
                if not project_id:
                    raise ConversationServiceError(
                        "No project associated with this conversation"
                    )

                async for chunk in self._generate_and_stream_ai_response(
                    message.content, conversation_id, user_id, message.node_ids
                ):
                    yield chunk

        except AccessTypeReadError:
            raise
        except Exception as e:
            logger.error(
                f"Error in store_message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to store message or generate AI response."
            ) from e

    async def _get_conversation_with_message_count(
        self, conversation_id: str
    ) -> Conversation:
        result = (
            self.sql_db.query(
                Conversation,
                func.count(Message.id)
                .filter(Message.type == MessageType.HUMAN)
                .label("human_message_count"),
            )
            .outerjoin(Message, Conversation.id == Message.conversation_id)
            .filter(Conversation.id == conversation_id)
            .group_by(Conversation.id)
            .first()
        )

        if result:
            conversation, human_message_count = result
            setattr(conversation, "human_message_count", human_message_count)
            return conversation
        return None

    async def _generate_title(
        self, conversation: Conversation, message_content: str
    ) -> str:
        agent_type = conversation.agent_ids[0]

        llm = await self.provider_service.get_small_llm(agent_type=AgentType.LANGCHAIN)
        prompt = ChatPromptTemplate.from_template(
            "Given an agent type '{agent_type}' and an initial message '{message}', "
            "generate a concise and relevant title for a conversation. "
            "The title should be no longer than 50 characters. Only return title string, do not wrap in quotes."
        )

        messages = prompt.format_messages(
            agent_type=agent_type, message=message_content
        )
        response = await llm.agenerate([messages])

        generated_title = response.generations[0][0].text.strip()
        if len(generated_title) > 50:
            generated_title = generated_title[:50].strip() + "..."
        return generated_title

    async def _update_conversation_title(self, conversation_id: str, new_title: str):
        self.sql_db.query(Conversation).filter_by(id=conversation_id).update(
            {"title": new_title, "updated_at": datetime.now(timezone.utc)}
        )
        self.sql_db.commit()

    async def regenerate_last_message(
        self, conversation_id: str, user_id: str, node_ids: List[NodeContext] = []
    ) -> AsyncGenerator[str, None]:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")
            last_human_message = await self._get_last_human_message(conversation_id)
            if not last_human_message:
                raise MessageNotFoundError("No human message found to regenerate from")

            await self._archive_subsequent_messages(
                conversation_id, last_human_message.created_at
            )
            PostHogClient().send_event(
                user_id,
                "regenerate_conversation_event",
                {"conversation_id": conversation_id},
            )

            async for chunk in self._generate_and_stream_ai_response(
                last_human_message.content, conversation_id, user_id, node_ids
            ):
                yield chunk
        except AccessTypeReadError:
            raise
        except MessageNotFoundError as e:
            logger.warning(
                f"No message to regenerate in conversation {conversation_id}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Error in regenerate_last_message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError("Failed to regenerate last message.") from e

    async def _get_last_human_message(self, conversation_id: str):
        message = (
            self.sql_db.query(Message)
            .filter_by(conversation_id=conversation_id, type=MessageType.HUMAN)
            .order_by(Message.created_at.desc())
            .first()
        )
        if not message:
            logger.warning(f"No human message found in conversation {conversation_id}")
        return message

    async def _archive_subsequent_messages(
        self, conversation_id: str, timestamp: datetime
    ):
        try:
            self.sql_db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.created_at > timestamp,
            ).update(
                {Message.status: MessageStatus.ARCHIVED}, synchronize_session="fetch"
            )
            self.sql_db.commit()
            logger.info(
                f"Archived subsequent messages in conversation {conversation_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to archive messages in conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to archive subsequent messages."
            ) from e

    async def _generate_and_stream_ai_response(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        conversation = (
            self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
        )
        if not conversation:
            raise ConversationNotFoundError(
                f"Conversation with id {conversation_id} not found"
            )

        agent_id = conversation.agent_ids[0]
        project_id = conversation.project_ids[0] if conversation.project_ids else None
        supervisor = SimplifiedAgentSupervisor(self.sql_db, self.provider_service)
        await supervisor.initialize(user_id)
        try:
            agent = self.agent_injector_service.get_agent(agent_id)

            logger.info(
                f"conversation_id: {conversation_id} Running agent {agent_id} with query: {query}"
            )

            if isinstance(agent, CustomAgentsService):
                # Custom agent doesn't support streaming, so we'll yield the entire response at once
                response = await agent.run(
                    agent_id, query, project_id, user_id, conversation.id, node_ids
                )
                yield response
            else:
                # For other agents that support streaming
                async for chunk in supervisor.process_query(
                    query, project_id, conversation.id, user_id, node_ids, agent_id
                ):
                    yield chunk

            logger.info(
                f"Generated and streamed AI response for conversation {conversation.id} for user {user_id} using agent {agent_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to generate and stream AI response for conversation {conversation.id}: {e}",
                exc_info=True,
            )
            raise ConversationServiceError(
                "Failed to generate and stream AI response."
            ) from e

    async def delete_conversation(self, conversation_id: str, user_id: str) -> dict:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")
            # Use a nested transaction if one is already in progress
            with self.sql_db.begin_nested():
                # Delete related messages first
                deleted_messages = (
                    self.sql_db.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .delete(synchronize_session="fetch")
                )

                deleted_conversation = (
                    self.sql_db.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .delete(synchronize_session="fetch")
                )

                if deleted_conversation == 0:
                    raise ConversationNotFoundError(
                        f"Conversation with id {conversation_id} not found"
                    )

            # If we get here, commit the transaction
            self.sql_db.commit()

            PostHogClient().send_event(
                user_id,
                "delete_conversation_event",
                {"conversation_id": conversation_id},
            )

            logger.info(
                f"Deleted conversation {conversation_id} and {deleted_messages} related messages"
            )
            return {
                "status": "success",
                "message": f"Conversation {conversation_id} and its messages have been permanently deleted.",
                "deleted_messages_count": deleted_messages,
            }

        except ConversationNotFoundError as e:
            logger.warning(str(e))
            self.sql_db.rollback()
            raise
        except AccessTypeReadError:
            raise

        except SQLAlchemyError as e:
            logger.error(f"Database error in delete_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to a database error"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in delete_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to an unexpected error"
            ) from e

    async def get_conversation_info(
        self, conversation_id: str, user_id: str
    ) -> ConversationInfoResponse:
        try:
            conversation = (
                self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )
            is_creator = conversation.user_id == user_id
            access_type = await self.check_conversation_access(
                conversation_id, self.user_email
            )

            if access_type == ConversationAccessType.NOT_FOUND:
                raise AccessTypeNotFoundError("Access type not found")

            total_messages = (
                self.sql_db.query(Message)
                .filter_by(conversation_id=conversation_id, status=MessageStatus.ACTIVE)
                .count()
            )
            return ConversationInfoResponse(
                id=conversation.id,
                title=conversation.title,
                status=conversation.status,
                project_ids=conversation.project_ids,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                total_messages=total_messages,
                agent_ids=conversation.agent_ids,
                access_type=access_type,
                is_creator=is_creator,
                creator_id=conversation.user_id,
            )
        except ConversationNotFoundError as e:
            logger.warning(str(e))
            raise
        except AccessTypeNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error in get_conversation_info: {e}", exc_info=True)
            raise ConversationServiceError(
                f"Failed to get conversation info for {conversation_id}"
            ) from e

    async def get_conversation_messages(
        self, conversation_id: str, start: int, limit: int, user_id: str
    ) -> List[MessageResponse]:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email
            )
            if access_level == ConversationAccessType.NOT_FOUND:
                raise AccessTypeNotFoundError("Access denied.")
            conversation = (
                self.sql_db.query(Conversation).filter_by(id=conversation_id).first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            messages = (
                self.sql_db.query(Message)
                .filter_by(conversation_id=conversation_id)
                .filter_by(status=MessageStatus.ACTIVE)
                .filter(Message.type != MessageType.SYSTEM_GENERATED)
                .order_by(Message.created_at)
                .offset(start)
                .limit(limit)
                .all()
            )

            return [
                MessageResponse(
                    id=message.id,
                    conversation_id=message.conversation_id,
                    content=message.content,
                    sender_id=message.sender_id,
                    type=message.type,
                    status=message.status,
                    created_at=message.created_at,
                    citations=(
                        message.citations.split(",") if message.citations else None
                    ),
                )
                for message in messages
            ]
        except ConversationNotFoundError as e:
            logger.warning(str(e))
            raise
        except AccessTypeNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error in get_conversation_messages: {e}", exc_info=True)
            raise ConversationServiceError(
                f"Failed to get messages for conversation {conversation_id}"
            ) from e

    async def stop_generation(self, conversation_id: str, user_id: str) -> dict:
        logger.info(f"Attempting to stop generation for conversation {conversation_id}")
        return {"status": "success", "message": "Generation stop request received"}

    async def rename_conversation(
        self, conversation_id: str, new_title: str, user_id: str
    ) -> dict:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")
            conversation = (
                self.sql_db.query(Conversation)
                .filter_by(id=conversation_id, user_id=user_id)
                .first()
            )
            if not conversation:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            conversation.title = new_title
            conversation.updated_at = datetime.now(timezone.utc)
            self.sql_db.commit()

            logger.info(
                f"Renamed conversation {conversation_id} to '{new_title}' by user {user_id}"
            )
            return {
                "status": "success",
                "message": f"Conversation renamed to '{new_title}'",
            }

        except SQLAlchemyError as e:
            logger.error(f"Database error in rename_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to rename conversation due to a database error"
            ) from e
        except AccessTypeReadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in rename_conversation: {e}", exc_info=True)
            self.sql_db.rollback()
            raise ConversationServiceError(
                "Failed to rename conversation due to an unexpected error"
            ) from e
