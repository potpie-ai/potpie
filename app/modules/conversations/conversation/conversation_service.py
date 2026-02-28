import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, List, Optional, Dict, Union
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.utils.logger import setup_logger
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
    Visibility,
)
from app.modules.conversations.conversation.conversation_schema import (
    ChatMessageResponse,
    ConversationAccessType,
    ConversationInfoResponse,
    CreateConversationRequest,
)
from app.modules.conversations.message.message_model import (
    MessageType,
)
from app.modules.conversations.message.message_schema import (
    MessageRequest,
    MessageResponse,
    NodeContext,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_model import (
    CustomAgent as CustomAgentModel,
)
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentService,
)
from app.modules.users.user_schema import UserConversationListResponse
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.intelligence.agents.context_config import (
    ESTIMATED_TOKENS_PER_MESSAGE,
    HISTORY_MESSAGE_CAP,
    get_history_token_budget,
)
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager.sync_helper import ensure_repo_registered
from app.modules.users.user_service import UserService
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.media.media_service import MediaService
from app.modules.conversations.session.session_service import SessionService
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.celery.celery_app import celery_app
from app.modules.conversations.exceptions import GenerationCancelled
from .conversation_store import ConversationStore, StoreError
from ..message.message_store import MessageStore

logger = setup_logger(__name__)


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


class ConversationService:
    def __init__(
        self,
        db: Session,
        user_id: str,
        user_email: str,
        conversation_store: ConversationStore,
        message_store: MessageStore,
        project_service: ProjectService,
        history_manager: ChatHistoryService,
        provider_service: ProviderService,
        tools_service: ToolService,
        promt_service: PromptService,
        agent_service: AgentsService,
        custom_agent_service: CustomAgentService,
        media_service: MediaService,
        session_service: SessionService = None,
        redis_manager: RedisStreamManager = None,
    ):
        self.db = db
        self.user_id = user_id
        self.user_email = user_email
        self.conversation_store = conversation_store
        self.message_store = message_store
        self.project_service = project_service
        self.history_manager = history_manager
        self.provider_service = provider_service
        self.tool_service = tools_service
        self.prompt_service = promt_service
        self.agent_service = agent_service
        self.custom_agent_service = custom_agent_service
        self.media_service = media_service
        # Dependency injection for stop_generation
        self.session_service = session_service or SessionService()
        self.redis_manager = redis_manager or RedisStreamManager()
        self.celery_app = celery_app

        # Initialize repo manager if enabled
        self.repo_manager = None
        try:
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if repo_manager_enabled:
                from app.modules.repo_manager import RepoManager

                self.repo_manager = RepoManager()
                logger.info("ConversationService: RepoManager initialized")
        except Exception as e:
            logger.warning(
                f"ConversationService: Failed to initialize RepoManager: {e}"
            )

    @classmethod
    def create(
        cls,
        conversation_store: ConversationStore,
        message_store: MessageStore,
        db: Session,
        user_id: str,
        user_email: str,
    ):
        project_service = ProjectService(db)
        history_manager = ChatHistoryService(db)
        provider_service = ProviderService(db, user_id)
        tool_service = ToolService(db, user_id)
        prompt_service = PromptService(db)
        agent_service = AgentsService(
            db, provider_service, prompt_service, tool_service
        )
        custom_agent_service = CustomAgentService(db, provider_service, tool_service)
        media_service = MediaService(db)
        session_service = SessionService()
        redis_manager = RedisStreamManager()

        return cls(
            db,
            user_id,
            user_email,
            conversation_store,
            message_store,
            project_service,
            history_manager,
            provider_service,
            tool_service,
            prompt_service,
            agent_service,
            custom_agent_service,
            media_service,
            session_service,
            redis_manager,
        )

    async def check_conversation_access(
        self, conversation_id: str, user_email: str, firebase_user_id: str = None
    ) -> str:
        if not user_email:
            return ConversationAccessType.WRITE

        # Use Firebase user ID directly if available, otherwise fall back to email lookup
        if firebase_user_id:
            user_id = firebase_user_id
        else:
            user_service = UserService(self.sql_db)
            user_id = user_service.get_user_id_by_email(user_email)

        # Retrieve the conversation
        conversation = await self.conversation_store.get_by_id(conversation_id)

        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found in database")
            return (
                ConversationAccessType.NOT_FOUND
            )  # Return 'not found' if conversation doesn't exist

        if not conversation.visibility:
            conversation.visibility = Visibility.PRIVATE

        if user_id == conversation.user_id:  # Check if the user is the creator
            return ConversationAccessType.WRITE  # Creator always has write access

        if conversation.visibility == Visibility.PUBLIC:
            return ConversationAccessType.READ  # Public users get read access

        # Check if the conversation is shared
        if conversation.shared_with_emails:
            user_service = UserService(self.sql_db)
            shared_user_ids = user_service.get_user_ids_by_emails(
                conversation.shared_with_emails
            )
            if shared_user_ids is None:
                logger.warning(
                    "Failed to get user IDs for shared emails, returning NOT_FOUND"
                )
                return ConversationAccessType.NOT_FOUND
            # Check if the current user ID is in the shared user IDs
            if user_id in shared_user_ids:
                return ConversationAccessType.READ  # Shared users can only read
            else:
                return ConversationAccessType.NOT_FOUND

        return ConversationAccessType.NOT_FOUND

    async def create_conversation(
        self,
        conversation: CreateConversationRequest,
        user_id: str,
        hidden: bool = False,
        local_mode: bool = False,
    ) -> tuple[str, str]:
        try:
            if not await self.agent_service.validate_agent_id(
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

            conversation_id = await self._create_conversation_record(
                conversation, title, user_id, hidden
            )

            # Fetch project structure in background with timeout and error handling
            # This is fire-and-forget to avoid blocking conversation creation
            async def _fetch_structure_with_timeout():
                try:
                    # Add timeout to prevent hanging on large repositories
                    # Note: This may not interrupt synchronous blocking calls, but will
                    # prevent the task from running indefinitely
                    await asyncio.wait_for(
                        CodeProviderService(self.db).get_project_structure_async(
                            conversation.project_ids[0]
                        ),
                        timeout=30.0,  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout fetching project structure for project {conversation.project_ids[0]}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error fetching project structure for project {conversation.project_ids[0]}: {e}",
                        exc_info=True,
                    )

            # Create background task with proper exception handling
            fetch_task = asyncio.create_task(_fetch_structure_with_timeout())

            def _on_fetch_done(t: asyncio.Task) -> None:
                if t.cancelled():
                    return
                try:
                    exc = t.exception()
                except asyncio.CancelledError:
                    return
                if exc is not None:
                    logger.exception("Failed to fetch project structure", exc_info=exc)

            fetch_task.add_done_callback(_on_fetch_done)

            # Ensure repo is registered and cloned in repo manager (skip in local/VSCode mode)
            if not local_mode and self.repo_manager and conversation.project_ids:
                project_id_str = str(conversation.project_ids[0])
                # Fast path: register any existing local copy (5s timeout)
                asyncio.create_task(
                    self._ensure_repo_in_repo_manager(project_id_str, user_id)
                )
                # Slow path: clone the repo if it's missing entirely (no timeout — runs until done)
                asyncio.create_task(
                    self._clone_repo_if_missing(project_id_str, user_id)
                )

            await self._add_system_message(conversation_id, project_name, user_id)

            return conversation_id, "Conversation created successfully."
        except IntegrityError as e:
            logger.exception("IntegrityError in create_conversation", user_id=user_id)
            raise ConversationServiceError(
                "Failed to create conversation due to a database integrity error."
            ) from e
        except Exception as e:
            logger.exception("Unexpected error in create_conversation", user_id=user_id)
            raise ConversationServiceError(
                "An unexpected error occurred while creating the conversation."
            ) from e

    async def _create_conversation_record(
        self,
        conversation: CreateConversationRequest,
        title: str,
        user_id: str,
        hidden: bool = False,
    ) -> str:
        conversation_id = str(uuid7())
        new_conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title,
            status=ConversationStatus.ARCHIVED if hidden else ConversationStatus.ACTIVE,
            project_ids=conversation.project_ids,
            agent_ids=conversation.agent_ids,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await self.conversation_store.create(new_conversation)

        logger.info(
            f"Project id : {conversation.project_ids[0]} Created new conversation with ID: {conversation_id}, title: {title}, user_id: {user_id}, agent_id: {conversation.agent_ids[0]}, hidden: {hidden}"
        )
        return conversation_id

    async def _ensure_repo_in_repo_manager(self, project_id: str, user_id: str) -> None:
        """
        Ensure that the repository for a project is registered in the repo manager.
        If the repo doesn't exist, attempts to register it if the project has been parsed.

        This runs in a thread pool to avoid blocking the async event loop with filesystem operations.

        Args:
            project_id: The project ID
            user_id: The user ID
        """
        if not self.repo_manager:
            return  # Repo manager not enabled

        # Run filesystem operations in a thread pool to avoid blocking
        # Add timeout to prevent hanging
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._ensure_repo_in_repo_manager_sync, project_id, user_id
                ),
                timeout=5.0,  # 5 second timeout to prevent hanging
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout ensuring repo in repo manager for project {project_id} (took >5s)"
            )
        except Exception as e:
            logger.warning(
                f"Error ensuring repo in repo manager for project {project_id}: {e}",
                exc_info=True,
            )
            # Don't fail the message if repo registration fails

    def _ensure_repo_in_repo_manager_sync(self, project_id: str, user_id: str) -> None:
        """
        Synchronous version of _ensure_repo_in_repo_manager.
        Runs in a thread pool to avoid blocking the async event loop.
        Delegates to shared ensure_repo_registered helper.
        """
        if not self.repo_manager:
            return

        try:
            try:
                project = self.project_service.get_project_from_db_by_id_sync(
                    project_id
                )  # type: ignore[arg-type]
            except (TypeError, ValueError):
                try:
                    project = self.project_service.get_project_from_db_by_id_sync(
                        int(project_id)
                    )  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    logger.warning(
                        f"Cannot ensure repo in repo manager: invalid project_id {project_id}"
                    )
                    return

            if not project:
                logger.warning(
                    f"Cannot ensure repo in repo manager: project {project_id} not found"
                )
                return

            # Map project keys to expected format (project uses project_name, etc.)
            project_data = {
                "project_name": project.get("project_name"),
                "branch_name": project.get("branch_name"),
                "commit_id": project.get("commit_id"),
                "repo_path": project.get("repo_path"),
                "status": project.get("status"),
            }
            ensure_repo_registered(
                project_data,
                user_id,
                self.repo_manager,
                registered_from="conversation_message",
            )
        except Exception as e:
            logger.warning(
                f"Error in _ensure_repo_in_repo_manager_sync for project {project_id}: {e}",
                exc_info=True,
            )

    def _needs_full_clone_sync(self, project_id: str) -> bool:
        """
        Returns True only when a full git clone is needed (bare repo not on disk yet).
        When the bare repo already exists, prepare_for_parsing just creates a worktree
        quickly — no loading message is needed in that case.
        On any error or local repo, returns False so we never show a spurious message.
        """
        if not self.repo_manager:
            return False
        try:
            try:
                project = self.project_service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                try:
                    project = self.project_service.get_project_from_db_by_id_sync(int(project_id))  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    return False
            if not project:
                return False
            repo_name = project.get("project_name")
            repo_path = project.get("repo_path")
            if not repo_name or repo_path:
                return False  # local repo, always considered available
            # Check if the bare repo directory exists on disk.
            # If it does, worktree creation is fast — no loading message needed.
            bare_repo_path = self.repo_manager._get_bare_repo_path(repo_name)  # noqa: SLF001
            return not bare_repo_path.exists()
        except Exception:
            logger.warning(
                f"Error checking bare repo for project {project_id}, assuming available",
                exc_info=True,
            )
            return False

    def _clone_repo_if_missing_sync(self, project_id: str, user_id: str) -> None:
        """
        Ensure the repo worktree exists in the repo manager.
        Calls prepare_for_parsing if the repo is not yet registered or the worktree
        is missing. Fast when the bare repo already exists (just adds the worktree);
        slow only on the very first clone. Safe to call on every message.
        """
        if not self.repo_manager:
            logger.info(f"[clone_sync] Skipping project {project_id}: no repo_manager")
            return
        try:
            try:
                project = self.project_service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                try:
                    project = self.project_service.get_project_from_db_by_id_sync(int(project_id))  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    logger.warning(f"[clone_sync] Cannot parse project_id={project_id}")
                    return

            if not project:
                logger.warning(f"[clone_sync] Project {project_id} not found in DB")
                return

            repo_name = project.get("project_name")
            branch = project.get("branch_name")
            commit_id = project.get("commit_id")
            repo_path = project.get("repo_path")

            logger.info(
                f"[clone_sync] project={project_id} repo_name={repo_name!r} "
                f"branch={branch!r} commit_id={commit_id!r} repo_path={repo_path!r}"
            )

            if not repo_name:
                logger.warning(f"[clone_sync] Skipping project {project_id}: no repo_name")
                return
            if repo_path:
                logger.info(
                    f"[clone_sync] Skipping project {project_id}: has local repo_path={repo_path!r}"
                )
                return

            # Match the same ref priority used by get_file_content: commit_id first,
            # then branch. This ensures the worktree key registered here is the same key
            # that all code-provider tools use for their lookups.
            ref = commit_id if commit_id else branch
            is_commit_ref = bool(commit_id)

            available = self.repo_manager.is_repo_available(
                repo_name, branch=branch, commit_id=commit_id, user_id=user_id
            )
            logger.info(
                f"[clone_sync] is_repo_available({repo_name!r}@{ref!r}) = {available}"
            )
            if available:
                return  # already there

            if not ref:
                logger.warning(f"[clone_sync] Skipping {repo_name}: no ref (branch or commit_id)")
                return

            logger.info(
                f"[clone_sync] Calling prepare_for_parsing for {repo_name}@{ref} "
                f"(is_commit={is_commit_ref})"
            )
            from app.modules.code_provider.github.github_service import GithubService

            user_token = GithubService(self.db).get_github_oauth_token(user_id)

            # Try with user token first; fall back to env-var token (GH_TOKEN / GITHUB_TOKEN)
            # if the user token is missing or returns a permission error — matching the
            # same retry strategy used in parsing_helper.
            for attempt, token in enumerate([user_token, None]):
                if attempt == 1 and token == user_token:
                    break  # no point retrying with the same (None) token
                try:
                    self.repo_manager.prepare_for_parsing(
                        repo_name,
                        ref,
                        auth_token=token,
                        is_commit=is_commit_ref,
                        user_id=user_id,
                    )
                    logger.info(
                        f"[clone_sync] Worktree ready: {repo_name}@{ref} "
                        f"(attempt {attempt + 1}, token={'user' if token else 'env'})"
                    )
                    break
                except Exception as e:
                    if attempt == 0 and token is not None:
                        logger.warning(
                            f"[clone_sync] User-token clone failed for {repo_name}@{ref}: {e}. "
                            "Retrying with env token.",
                        )
                    else:
                        logger.warning(
                            f"[clone_sync] prepare_for_parsing failed for {repo_name}@{ref}: {e}",
                            exc_info=True,
                        )
        except Exception as e:
            logger.warning(
                f"[clone_sync] Unexpected error for project {project_id}: {e}",
                exc_info=True,
            )

    async def _clone_repo_if_missing(self, project_id: str, user_id: str) -> None:
        """
        Fire-and-forget background clone. Runs in a thread so it never blocks the
        event loop, with no timeout — the clone runs until it completes or fails.
        """
        if not self.repo_manager:
            return
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._clone_repo_if_missing_sync, project_id, user_id
            )
        except Exception as e:
            logger.warning(
                f"_clone_repo_if_missing error for project {project_id}: {e}",
                exc_info=True,
            )

    async def _add_system_message(
        self, conversation_id: str, project_name: str, user_id: str
    ):
        content = f"You can now ask questions about the {project_name} repository."
        try:
            await self.message_store.create_system_message(conversation_id, content)
            logger.info(
                f"Added system message to conversation {conversation_id} for user {user_id}"
            )
        except Exception as e:
            logger.exception(
                f"Failed to add system message to conversation {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
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
        stream: bool = True,
        local_mode: bool = False,
        run_id: Optional[str] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
    ) -> AsyncGenerator[ChatMessageResponse, None]:
        try:
            logger.info(
                f"DEBUG: store_message called with message.attachment_ids: {message.attachment_ids}"
            )
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email, user_id
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")
            self.history_manager.add_message_chunk(
                conversation_id, message.content, message_type, user_id
            )
            message_id = self.history_manager.flush_message_buffer(
                conversation_id, message_type, user_id
            )
            logger.info(f"Stored message in conversation {conversation_id}")

            # Handle attachments if present
            if message_type == MessageType.HUMAN and message.attachment_ids:
                try:
                    await self.media_service.update_message_attachments(
                        message_id, message.attachment_ids
                    )
                    logger.info(
                        f"Linked {len(message.attachment_ids)} attachments to message {message_id}"
                    )
                except Exception:
                    logger.exception(
                        f"Failed to link attachments to message {message_id}",
                        message_id=message_id,
                        conversation_id=conversation_id,
                    )
                    # Continue processing even if attachment linking fails

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

                # Ensure repo is registered in repo manager (skip in local/VSCode mode)
                # Convert project_id to string if needed (it might be a Column object)
                project_id_str = str(project_id) if project_id else None
                if project_id_str and not local_mode and self.repo_manager:
                    logger.info(
                        f"[store_message] Checking/creating worktree for project {project_id_str}"
                    )
                    # Only show loading message when a full clone is needed (bare repo missing).
                    # When the bare repo already exists, worktree creation is fast — no message.
                    needs_clone = await asyncio.get_event_loop().run_in_executor(
                        None, self._needs_full_clone_sync, project_id_str
                    )
                    logger.info(
                        f"[store_message] needs_full_clone={needs_clone} for project {project_id_str}"
                    )
                    if needs_clone and stream:
                        yield ChatMessageResponse(
                            message="⏳ Setting up repository workspace, please wait...",
                            citations=[],
                            tool_calls=[],
                        )
                    # Always run synchronously: creates bare repo + worktree if missing,
                    # or returns quickly if everything already exists.
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._clone_repo_if_missing_sync,
                        project_id_str,
                        user_id,
                    )
                elif project_id_str and not local_mode:
                    logger.info(
                        f"[store_message] No repo_manager, skipping worktree setup for project {project_id_str}"
                    )
                    await self._ensure_repo_in_repo_manager(project_id_str, user_id)

                logger.info(
                    f"[store_message] message.tunnel_url={message.tunnel_url}, "
                    f"conversation_id={conversation_id}, user_id={user_id}"
                )
                if stream:
                    async for chunk in self._generate_and_stream_ai_response(
                        message.content,
                        conversation_id,
                        user_id,
                        message.node_ids,
                        message.attachment_ids,
                        local_mode=local_mode,
                        tunnel_url=message.tunnel_url,
                        run_id=run_id,
                        check_cancelled=check_cancelled,
                    ):
                        yield chunk
                else:
                    full_message = ""
                    all_citations = []
                    accumulated_thinking = None
                    async for chunk in self._generate_and_stream_ai_response(
                        message.content,
                        conversation_id,
                        user_id,
                        message.node_ids,
                        message.attachment_ids,
                        local_mode=local_mode,
                        tunnel_url=message.tunnel_url,
                        run_id=run_id,
                        check_cancelled=check_cancelled,
                    ):
                        full_message += chunk.message
                        all_citations = all_citations + chunk.citations
                        if chunk.thinking:
                            accumulated_thinking = chunk.thinking

                    yield ChatMessageResponse(
                        message=full_message, citations=all_citations, tool_calls=[], thinking=accumulated_thinking
                    )

        except AccessTypeReadError:
            raise
        except GenerationCancelled:
            raise
        except Exception as e:
            logger.exception(
                f"Error in store_message for conversation {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                "Failed to store message or generate AI response."
            ) from e

    async def _get_conversation_with_message_count(
        self, conversation_id: str
    ) -> Conversation:
        return await self.conversation_store.get_with_message_count(conversation_id)

    async def _generate_title(
        self, conversation: Conversation, message_content: str
    ) -> str:
        agent_type = conversation.agent_ids[0]

        prompt = (
            "Given an agent type '{agent_type}' and an initial message '{message}', "
            "generate a concise and relevant title for a conversation. "
            "The title should be no longer than 50 characters. Only return title string, do not wrap in quotes."
        ).format(agent_type=agent_type, message=message_content)

        messages = [
            {
                "role": "system",
                "content": "You are a conversation title generator that creates concise and relevant titles.",
            },
            {"role": "user", "content": prompt},
        ]
        generated_title: str = await self.provider_service.call_llm(
            messages=messages, config_type="chat"
        )  # type: ignore

        if len(generated_title) > 50:
            generated_title = generated_title[:50].strip() + "..."
        return generated_title

    async def _update_conversation_title(self, conversation_id: str, new_title: str):
        await self.conversation_store.update_title(conversation_id, new_title)

    async def regenerate_last_message(
        self,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext] = [],
        stream: bool = True,
        local_mode: bool = False,
    ) -> AsyncGenerator[ChatMessageResponse, None]:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email, user_id
            )
            if access_level != ConversationAccessType.WRITE:
                raise AccessTypeReadError(
                    "Access denied. Only conversation creators can regenerate messages."
                )
            last_human_message = await self._get_last_human_message(conversation_id)
            if not last_human_message:
                raise MessageNotFoundError("No human message found to regenerate from")

            # Get attachment IDs from the last human message
            attachment_ids = None
            if last_human_message.has_attachments:
                try:
                    attachments = await self.media_service.get_message_attachments(
                        last_human_message.id, include_download_urls=False
                    )
                    # Extract only image attachment IDs for multimodal processing
                    from app.modules.media.media_model import AttachmentType

                    attachment_ids = [
                        att.id
                        for att in attachments
                        if att.attachment_type == AttachmentType.IMAGE
                    ]
                    if attachment_ids:
                        logger.info(
                            f"Found {len(attachment_ids)} image attachments for regeneration: {attachment_ids}"
                        )
                    else:
                        logger.info("No image attachments found in last human message")
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve attachments for message {last_human_message.id}: {e}"
                    )
                    attachment_ids = None

            await self._archive_subsequent_messages(
                conversation_id, last_human_message.created_at
            )
            PostHogClient().send_event(
                user_id,
                "regenerate_conversation_event",
                {"conversation_id": conversation_id},
            )

            if stream:
                async for chunk in self._generate_and_stream_ai_response(
                    last_human_message.content,
                    conversation_id,
                    user_id,
                    node_ids,
                    attachment_ids,
                    local_mode=local_mode,
                    tunnel_url=None,
                ):
                    yield chunk
            else:
                full_message = ""
                all_citations = []
                accumulated_thinking = None

                async for chunk in self._generate_and_stream_ai_response(
                    last_human_message.content,
                    conversation_id,
                    user_id,
                    node_ids,
                    attachment_ids,
                    local_mode=local_mode,
                    tunnel_url=None,
                ):
                    full_message += chunk.message
                    all_citations = all_citations + chunk.citations
                    if chunk.thinking:
                        accumulated_thinking = chunk.thinking

                yield ChatMessageResponse(
                    message=full_message, citations=all_citations, tool_calls=[], thinking=accumulated_thinking
                )

        except AccessTypeReadError:
            raise
        except MessageNotFoundError as e:
            logger.warning(
                f"No message to regenerate in conversation {conversation_id}: {e}"
            )
            raise
        except Exception as e:
            logger.exception(
                f"Error in regenerate_last_message for conversation {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError("Failed to regenerate last message.") from e

    async def regenerate_last_message_background(
        self,
        conversation_id: str,
        node_ids: Optional[List[str]] = None,
        attachment_ids: List[str] = [],
        local_mode: bool = False,
        run_id: Optional[str] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
    ) -> AsyncGenerator[ChatMessageResponse, None]:
        """Background version of regenerate_last_message for Celery task execution"""
        try:
            # Access control validation
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email, self.user_id
            )
            if access_level != ConversationAccessType.WRITE:
                raise AccessTypeReadError(
                    "Access denied. Only conversation creators can regenerate messages."
                )

            # Get last human message (already validated by background task caller)
            last_human_message = await self._get_last_human_message(conversation_id)
            if not last_human_message:
                raise MessageNotFoundError("No human message found to regenerate from")

            # Archive subsequent messages
            await self._archive_subsequent_messages(
                conversation_id, last_human_message.created_at
            )

            # PostHog analytics
            PostHogClient().send_event(
                self.user_id,
                "regenerate_conversation_event",
                {"conversation_id": conversation_id},
            )

            # Convert string node_ids to NodeContext objects for compatibility
            node_contexts = []
            if node_ids:
                node_contexts = [NodeContext(node_id=node_id) for node_id in node_ids]

            # Execute AI response generation with existing logic
            async for chunk in self._generate_and_stream_ai_response(
                last_human_message.content,
                conversation_id,
                self.user_id,
                node_contexts,
                attachment_ids,
                tunnel_url=None,
                local_mode=local_mode,
                run_id=run_id,
                check_cancelled=check_cancelled,
            ):
                yield chunk

        except GenerationCancelled:
            raise
        except (AccessTypeReadError, MessageNotFoundError):
            logger.exception(
                f"Background regeneration error for {conversation_id}",
                conversation_id=conversation_id,
                user_id=self.user_id,
            )
            raise
        except Exception as e:
            logger.exception(
                f"Background regeneration failed for {conversation_id}",
                conversation_id=conversation_id,
                user_id=self.user_id,
            )
            raise ConversationServiceError(f"Failed to regenerate message: {str(e)}")

    async def _get_last_human_message(self, conversation_id: str):
        message = await self.message_store.get_last_human_message(conversation_id)

        if not message:
            logger.warning(f"No human message found in conversation {conversation_id}")
        return message

    async def _archive_subsequent_messages(
        self, conversation_id: str, timestamp: datetime
    ):
        try:
            await self.message_store.archive_messages_after(conversation_id, timestamp)

            logger.info(
                f"Archived subsequent messages in conversation {conversation_id}"
            )
        except Exception as e:
            logger.exception(
                f"Failed to archive messages in conversation {conversation_id}",
                conversation_id=conversation_id,
                user_id=self.user_id,
            )
            raise ConversationServiceError(
                "Failed to archive subsequent messages."
            ) from e

    def parse_str_to_message(self, chunk: str) -> ChatMessageResponse:
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse chunk as JSON")
            raise ConversationServiceError("Failed to parse AI response") from e

        # Extract the 'message' and 'citations'
        message: str = data.get("message", "")
        citations: List[str] = data.get("citations", [])
        tool_calls: List[dict] = data.get("tool_calls", [])

        return ChatMessageResponse(
            message=message, citations=citations, tool_calls=tool_calls
        )

    async def _generate_and_stream_ai_response(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext],
        attachment_ids: Optional[List[str]] = None,
        local_mode: bool = False,
        tunnel_url: Optional[str] = None,
        run_id: Optional[str] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
    ) -> AsyncGenerator[ChatMessageResponse, None]:
        logger.info(
            f"[_generate_and_stream_ai_response] tunnel_url={tunnel_url}, "
            f"conversation_id={conversation_id}, user_id={user_id}, local_mode={local_mode}"
        )
        conversation = await self.conversation_store.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundError(
                f"Conversation with id {conversation_id} not found"
            )

        agent_id = conversation.agent_ids[0]
        project_id = conversation.project_ids[0] if conversation.project_ids else None

        try:
            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (f"{msg.type}: {msg.content}" if msg.content else msg)
                for msg in history
            ]

        except Exception:
            raise ConversationServiceError("Failed to get chat history")

        try:
            type = await self.agent_service.validate_agent_id(user_id, str(agent_id))
            if type is None:
                raise ConversationServiceError(f"Invalid agent_id {agent_id}")

            project_name = await self.project_service.get_project_name(
                project_ids=[project_id]
            )

            # Get project status to conditionally enable/disable tools
            project_info = await self.project_service.get_project_from_db_by_id(
                int(project_id)
                if isinstance(project_id, str) and project_id.isdigit()
                else project_id
            )
            project_status = project_info.get("status") if project_info else None

            # Prepare multimodal context - use current message attachments if available
            image_attachments = None
            if attachment_ids:
                image_attachments = await self._prepare_attachments_as_images(
                    attachment_ids
                )

            # Also get context images from recent conversation history
            context_images = await self._prepare_conversation_context_images(
                conversation_id
            )

            logger.info(
                f"conversation_id: {conversation_id} Running agent {agent_id} with query: {query}"
            )

            if image_attachments or context_images:
                logger.info(
                    f"Multimodal context: {len(image_attachments) if image_attachments else 0} current images, {len(context_images) if context_images else 0} context images"
                )

            # Single history cap for all agent types (Phase 2: token- and model-aware limits)
            token_budget = get_history_token_budget(None)
            msg_cap = min(
                HISTORY_MESSAGE_CAP,
                max(8, token_budget // ESTIMATED_TOKENS_PER_MESSAGE),
            )
            # Ensure we never pass empty history due to HISTORY_MESSAGE_CAP=0 or misconfig
            msg_cap = max(1, msg_cap)
            capped_history = validated_history[-msg_cap:]

            if type == "CUSTOM_AGENT":
                custom_ctx = ChatContext(
                    project_id=str(project_id),
                    project_name=project_name,
                    curr_agent_id=str(agent_id),
                    history=capped_history,
                    node_ids=[node.node_id for node in node_ids],
                    query=query,
                    project_status=project_status,
                    conversation_id=conversation_id,
                    user_id=user_id,  # Set user_id for tunnel routing
                    local_mode=local_mode,
                    repository=(
                        project_info.get("project_name")
                        if project_info
                        else project_name
                    ),
                    branch=project_info.get("branch_name") if project_info else None,
                )
                custom_ctx.check_cancelled = check_cancelled
                res = (
                    await self.agent_service.custom_agent_service.execute_agent_runtime(
                        user_id,
                        custom_ctx,
                    )
                )
                accumulated_tool_calls = []
                accumulated_thinking = None
                async for chunk in res:
                    if check_cancelled and check_cancelled():
                        raise GenerationCancelled()
                    # Accumulate tool_calls from each chunk
                    if chunk.tool_calls:
                        for tool_call in chunk.tool_calls:
                            tool_call_dict = (
                                tool_call.model_dump()
                                if hasattr(tool_call, "model_dump")
                                else tool_call.dict()
                                if hasattr(tool_call, "dict")
                                else tool_call
                            )
                            accumulated_tool_calls.append(tool_call_dict)
                    # Capture thinking content if present
                    if chunk.thinking:
                        accumulated_thinking = chunk.thinking
                    self.history_manager.add_message_chunk(
                        conversation_id,
                        chunk.response,
                        MessageType.AI_GENERATED,
                        citations=chunk.citations,
                        tool_calls=accumulated_tool_calls if chunk.tool_calls else None,
                    )
                    yield ChatMessageResponse(
                        message=chunk.response,
                        citations=chunk.citations,
                        tool_calls=[
                            tool_call.model_dump_json()
                            for tool_call in chunk.tool_calls
                        ],
                        thinking=chunk.thinking,
                    )
                # Flush with thinking content if captured
                self.history_manager.flush_message_buffer(
                    conversation_id, MessageType.AI_GENERATED, thinking=accumulated_thinking
                )
            else:
                # Create enhanced ChatContext with multimodal support
                nodes = [] if node_ids is None else [node.node_id for node in node_ids]
                chat_context = ChatContext(
                    project_id=str(project_id),
                    project_name=project_name,
                    curr_agent_id=str(agent_id),
                    history=capped_history,
                    node_ids=nodes,
                    query=query,
                    project_status=project_status,
                    image_attachments=image_attachments,
                    context_images=context_images,
                    conversation_id=conversation_id,
                    user_id=user_id,  # Set user_id for tunnel routing
                    tunnel_url=tunnel_url,  # Tunnel URL from request (takes priority)
                    local_mode=local_mode,
                    repository=(
                        project_info.get("project_name")
                        if project_info
                        else project_name
                    ),
                    branch=project_info.get("branch_name") if project_info else None,
                )
                chat_context.check_cancelled = check_cancelled

                res = self.agent_service.execute_stream(chat_context)

                accumulated_tool_calls = []
                accumulated_thinking = None
                async for chunk in res:
                    if check_cancelled and check_cancelled():
                        raise GenerationCancelled()
                    # Accumulate tool_calls from each chunk
                    if chunk.tool_calls:
                        for tool_call in chunk.tool_calls:
                            tool_call_dict = (
                                tool_call.model_dump()
                                if hasattr(tool_call, "model_dump")
                                else tool_call.dict()
                                if hasattr(tool_call, "dict")
                                else tool_call
                            )
                            accumulated_tool_calls.append(tool_call_dict)
                    # Capture thinking content if present
                    if chunk.thinking:
                        accumulated_thinking = chunk.thinking
                    self.history_manager.add_message_chunk(
                        conversation_id,
                        chunk.response,
                        MessageType.AI_GENERATED,
                        citations=chunk.citations,
                        tool_calls=accumulated_tool_calls if chunk.tool_calls else None,
                    )
                    yield ChatMessageResponse(
                        message=chunk.response,
                        citations=chunk.citations,
                        tool_calls=[
                            tool_call.model_dump_json()
                            for tool_call in chunk.tool_calls
                        ],
                        thinking=chunk.thinking,
                    )
                # Flush with thinking content if captured
                self.history_manager.flush_message_buffer(
                    conversation_id, MessageType.AI_GENERATED, thinking=accumulated_thinking
                )

            logger.info(
                f"Generated and streamed AI response for conversation {conversation.id} for user {user_id} using agent {agent_id}"
            )
        except GenerationCancelled:
            raise
        except Exception as e:
            logger.exception(
                f"Failed to generate and stream AI response for conversation {conversation.id}",
                conversation_id=conversation.id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                "Failed to generate and stream AI response."
            ) from e

    async def _generate_and_stream_ai_response_background(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        node_ids: List[NodeContext],
        attachment_ids: Optional[List[str]] = None,
        run_id: str = None,
    ) -> AsyncGenerator[ChatMessageResponse, None]:
        """Background version for Celery tasks - reuses existing streaming logic"""

        async for chunk in self._generate_and_stream_ai_response(
            query, conversation_id, user_id, node_ids, attachment_ids, tunnel_url=None
        ):
            yield chunk

    async def _prepare_attachments_as_images(
        self, attachment_ids: List[str]
    ) -> Optional[Dict[str, Dict[str, Union[str, int]]]]:
        """Convert attachment IDs directly to base64 images for multimodal processing"""
        try:
            if not attachment_ids:
                return None

            images = {}
            for attachment_id in attachment_ids:
                try:
                    # Get attachment info
                    attachment = await self.media_service.get_attachment(attachment_id)
                    logger.info(
                        f"DEBUG: Retrieved attachment {attachment_id}: type={attachment.attachment_type.value if attachment else 'None'}, mime_type={attachment.mime_type if attachment else 'None'}"
                    )
                    if (
                        attachment
                        and attachment.attachment_type.value.upper() == "IMAGE"
                    ):  # Check if it's an image
                        base64_data = await self.media_service.get_image_as_base64(
                            attachment_id
                        )
                        images[attachment_id] = {
                            "base64": base64_data,
                            "mime_type": attachment.mime_type,
                            "file_name": attachment.file_name,
                            "file_size": attachment.file_size,
                        }
                        logger.info(
                            f"Prepared image {attachment_id} ({attachment.file_name}) for multimodal processing"
                        )
                    else:
                        logger.info(
                            f"DEBUG: Skipping attachment {attachment_id} - not an image or attachment not found"
                        )
                except Exception:
                    logger.exception(
                        f"Failed to prepare attachment {attachment_id} as image",
                        attachment_id=attachment_id,
                    )
                    continue

            logger.info(
                f"Prepared {len(images)} images from {len(attachment_ids)} attachments for multimodal processing"
            )
            return images if images else None

        except Exception:
            logger.exception("Error preparing attachments as images")
            return None

    async def _prepare_current_message_images(
        self, conversation_id: str
    ) -> Optional[Dict[str, Dict[str, Union[str, int]]]]:
        """Get images from the most recent human message in the conversation"""
        try:
            # Get the most recent human message with attachments
            latest_human_message = (
                await self.message_store.get_latest_human_message_with_attachments(
                    conversation_id
                )
            )

            if not latest_human_message:
                return None

            # Get images from this message
            images = await self.media_service.get_message_images_as_base64(
                latest_human_message.id
            )
            return images if images else None

        except Exception:
            logger.exception(
                f"Error preparing current message images for conversation {conversation_id}",
                conversation_id=conversation_id,
            )
            return None

    async def _prepare_conversation_context_images(
        self, conversation_id: str, limit: int = 3
    ) -> Optional[Dict[str, Dict[str, Union[str, int]]]]:
        """Get recent images from conversation history for additional context"""
        try:
            # Get recent images from conversation (excluding the most recent message to avoid duplicates)
            context_images = await self.media_service.get_conversation_recent_images(
                conversation_id, limit=limit
            )
            return context_images if context_images else None

        except Exception:
            logger.exception(
                f"Error preparing conversation context images for conversation {conversation_id}",
                conversation_id=conversation_id,
            )
            return None

    async def delete_conversation(self, conversation_id: str, user_id: str) -> dict:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email, user_id
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")

            # Delete related messages first
            deleted_messages = await self.message_store.delete_for_conversation(
                conversation_id
            )

            # Delete the conversation
            deleted_conversation = await self.conversation_store.delete(conversation_id)

            if deleted_conversation == 0:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            # Phase 3: clear persisted compressed history for this conversation
            try:
                from app.modules.intelligence.agents.chat_agents.compressed_history_store import (
                    get_compressed_history_store,
                )

                store = get_compressed_history_store()
                if store:
                    store.delete(conversation_id, user_id=user_id)
            except Exception as e:
                logger.warning(
                    "Failed to delete compressed history for conversation %s: %s",
                    conversation_id,
                    e,
                    exc_info=False,
                )

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
            raise
        except AccessTypeReadError:
            raise
        except SQLAlchemyError as e:
            logger.exception(
                f"Database error in delete_conversation for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to a database error"
            ) from e
        except Exception as e:
            logger.exception(
                f"Unexpected error in delete_conversation for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"Failed to delete conversation {conversation_id} due to an unexpected error"
            ) from e

    async def get_conversation_info(
        self, conversation_id: str, user_id: str
    ) -> ConversationInfoResponse:
        try:
            logger.info("Getting info for conversation: {}", conversation_id)
            conversation = await self.conversation_store.get_by_id(conversation_id)

            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found in database")
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            is_creator = conversation.user_id == user_id

            access_type = await self.check_conversation_access(
                conversation_id, self.user_email, user_id
            )

            if access_type == ConversationAccessType.NOT_FOUND:
                logger.bind(conversation_id=conversation_id, user_id=user_id).error(
                    f"Access denied - access type is NOT_FOUND for user {user_id} on conversation {conversation_id}"
                )
                raise AccessTypeNotFoundError("Access type not found")

            total_messages = await self.message_store.count_active_for_conversation(
                conversation_id
            )

            agent_id = conversation.agent_ids[0] if conversation.agent_ids else None
            agent_ids = conversation.agent_ids
            if agent_id:
                system_agents = self.agent_service._system_agents(
                    self.provider_service, self.prompt_service, self.tool_service
                )

                if agent_id in system_agents.keys():
                    agent_ids = conversation.agent_ids
                else:
                    custom_agent = await self.custom_agent_service.get_agent_model(
                        agent_id
                    )

                    if custom_agent:
                        agent_ids = [custom_agent.role]

            result = ConversationInfoResponse(
                id=conversation.id,
                title=conversation.title,
                status=conversation.status,
                project_ids=conversation.project_ids,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                total_messages=total_messages,
                agent_ids=agent_ids,
                access_type=access_type,
                is_creator=is_creator,
                creator_id=conversation.user_id,
                visibility=conversation.visibility,
            )
            return result
        except ConversationNotFoundError as e:
            logger.warning(f"ConversationNotFoundError: {str(e)}")
            raise
        except AccessTypeNotFoundError:
            logger.exception(
                f"AccessTypeNotFoundError in get_conversation_info for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise
        except Exception as e:
            logger.exception(
                f"Error in get_conversation_info for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"Failed to get conversation info for {conversation_id}"
            ) from e

    async def get_conversation_messages(
        self, conversation_id: str, start: int, limit: int, user_id: str
    ) -> List[MessageResponse]:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email, user_id
            )

            if access_level == ConversationAccessType.NOT_FOUND:
                logger.bind(conversation_id=conversation_id, user_id=user_id).error(
                    f"Access denied - access level is NOT_FOUND for user {user_id} on conversation {conversation_id}"
                )
                raise AccessTypeNotFoundError("Access denied.")

            conversation = await self.conversation_store.get_by_id(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found in database")
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            messages = await self.message_store.get_active_for_conversation(
                conversation_id, start, limit
            )

            message_responses = []
            for message in messages:
                # Get attachments for this message
                attachments = None
                if message.has_attachments:
                    try:
                        attachments = await self.media_service.get_message_attachments(
                            message.id
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to get attachments for message {message.id}",
                            message_id=message.id,
                            conversation_id=conversation_id,
                        )
                        attachments = []

                message_responses.append(
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
                        has_attachments=message.has_attachments,
                        attachments=attachments,
                        tool_calls=message.tool_calls,
                        thinking=message.thinking,
                    )
                )
            return message_responses
        except ConversationNotFoundError as e:
            logger.warning(f"ConversationNotFoundError: {str(e)}")
            raise
        except AccessTypeNotFoundError:
            logger.exception(
                f"AccessTypeNotFoundError in get_conversation_messages for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise
        except Exception as e:
            logger.exception(
                f"Error in get_conversation_messages for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"Failed to get messages for conversation {conversation_id}"
            ) from e

    async def stop_generation(
        self, conversation_id: str, user_id: str, run_id: str = None
    ) -> dict:
        logger.info(
            f"Attempting to stop generation for conversation {conversation_id}, run_id: {run_id}"
        )

        # If run_id not provided, try to find active session
        if not run_id:
            from app.modules.conversations.conversation.conversation_schema import (
                ActiveSessionErrorResponse,
            )

            active_session = self.session_service.get_active_session(conversation_id)

            if isinstance(active_session, ActiveSessionErrorResponse):
                # No active session found - this is okay, just return success
                # The session might have already completed or been cleared
                logger.info(
                    f"No active session found for conversation {conversation_id} - already stopped or never started"
                )
                return {
                    "status": "success",
                    "message": "No active session to stop",
                }

            run_id = active_session.sessionId
            logger.info(
                f"Found active session {run_id} for conversation {conversation_id}"
            )

        # Retrieve task_id before any mutation so we know whether we will revoke (and thus need to save from stream).
        task_id = self.redis_manager.get_task_id(conversation_id, run_id)
        logger.info(
            f"Stop generation: conversation_id={conversation_id}, run_id={run_id}, task_id={task_id or 'none'}"
        )

        # Snapshot the stream before revoke so we have a consistent read (worker may be killed mid-write after revoke).
        snapshot = self.redis_manager.get_stream_snapshot(conversation_id, run_id)
        content_len = len(snapshot.get("content") or "")
        citations_count = len(snapshot.get("citations") or [])
        tool_calls_count = len(snapshot.get("tool_calls") or [])
        chunk_count = snapshot.get("chunk_count", 0)
        logger.info(
            f"Stream snapshot: conversation_id={conversation_id}, run_id={run_id}, "
            f"content_len={content_len}, citations={citations_count}, tool_calls={tool_calls_count}, chunk_count={chunk_count}"
        )

        # Set cancellation flag and revoke the Celery task so it stops producing chunks.
        self.redis_manager.set_cancellation(conversation_id, run_id)
        if task_id:
            try:
                # Step 1: Try graceful revocation first (cooperative cancellation).
                # The Celery task checks redis_manager.check_cancellation() regularly
                # and will exit cleanly when it sees the cancellation flag.
                self.celery_app.control.revoke(task_id, terminate=False)
                logger.info(
                    f"Sent graceful revoke for Celery task {task_id} for {conversation_id}:{run_id}"
                )

                # Step 2: Wait briefly for the task to stop gracefully
                time.sleep(0.5)

                # Check if task is still running via task status
                task_status = self.redis_manager.get_task_status(conversation_id, run_id)
                if task_status in ["running", "queued"]:
                    logger.info(
                        f"Task {task_id} still running after graceful revoke, using terminate"
                    )
                    # Step 3: Use terminate with SIGTERM as fallback
                    self.celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                    logger.info(
                        f"Sent SIGTERM to Celery task {task_id} for {conversation_id}:{run_id}"
                    )
                else:
                    logger.info(
                        f"Task {task_id} stopped gracefully for {conversation_id}:{run_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {task_id}: {str(e)}")
        else:
            logger.info(
                f"No task ID for {conversation_id}:{run_id} - already completed or revoked"
            )

        # Only save from stream when we revoked (worker did not flush). Persist content or tool-only placeholder.
        saved_partial = False
        saved_message_id = None
        if task_id:
            try:
                has_content = bool((snapshot.get("content") or "").strip())
                has_tool_calls = bool(snapshot.get("tool_calls"))
                if has_content:
                    content_to_save = (snapshot.get("content") or "").strip()
                elif has_tool_calls:
                    content_to_save = "(Generation stopped — tools were running.)"
                else:
                    content_to_save = ""
                if content_to_save:
                    saved_message_id = self.history_manager.save_partial_ai_message(
                        conversation_id,
                        content=content_to_save,
                        citations=snapshot.get("citations"),
                    )
                    saved_partial = saved_message_id is not None
                    if saved_partial:
                        logger.info(
                            f"Saved partial response for {conversation_id}:{run_id}, message_id={saved_message_id}"
                        )
                    else:
                        logger.info(
                            f"Save partial skipped for {conversation_id}:{run_id} (save_partial_ai_message returned None)"
                        )
                else:
                    logger.info(
                        f"Save partial skipped for {conversation_id}:{run_id}: no content and no tool_calls"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to save partial response on stop for {conversation_id}:{run_id}: {e}",
                    exc_info=True,
                )

        # Always clear the session - publish end event and update status
        # This ensures clients know the session is stopped and prevents stale sessions
        # This is important even if there's no task_id - it clears any stale session data
        # This will also handle the case where stop is called with a stale session_id
        try:
            self.redis_manager.clear_session(conversation_id, run_id)
        except Exception as e:
            logger.warning(
                f"Failed to clear session for {conversation_id}:{run_id}: {str(e)}"
            )
            # Continue anyway - the important part (revocation) is done

        return {
            "status": "success",
            "message": "Cancellation signal sent and task revoked",
            "saved_partial": saved_partial,
            "message_id": saved_message_id,
        }

    async def rename_conversation(
        self, conversation_id: str, new_title: str, user_id: str
    ) -> dict:
        try:
            access_level = await self.check_conversation_access(
                conversation_id, self.user_email, user_id
            )
            if access_level == ConversationAccessType.READ:
                raise AccessTypeReadError("Access denied.")

            conversation = await self.conversation_store.get_by_id(conversation_id)

            if not conversation or conversation.user_id != user_id:
                raise ConversationNotFoundError(
                    f"Conversation with id {conversation_id} not found"
                )

            await self.conversation_store.update_title(conversation_id, new_title)

            logger.info(
                f"Renamed conversation {conversation_id} to '{new_title}' by user {user_id}"
            )
            return {
                "status": "success",
                "message": f"Conversation renamed to '{new_title}'",
            }

        except SQLAlchemyError as e:
            logger.exception(
                f"Database error in rename_conversation for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                "Failed to rename conversation due to a database error"
            ) from e
        except AccessTypeReadError:
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error in rename_conversation for {conversation_id}",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ConversationServiceError(
                "Failed to rename conversation due to an unexpected error"
            ) from e

    async def get_conversations_with_projects_for_user(
        self,
        user_id: str,
        start: int,
        limit: int,
        sort: str = "updated_at",
        order: str = "desc",
    ) -> List[Conversation]:
        """
        Orchestrates the retrieval of conversations for a user by delegating to the store.
        """
        try:
            # The service's job is now just to call the store.
            # All the complex query logic is gone.
            return await self.conversation_store.get_for_user(
                user_id=user_id,
                start=start,
                limit=limit,
                sort=sort,
                order=order,
            )
        except StoreError as e:
            # Catch the specific error from the store and wrap it in a
            # service-level exception, which is a good practice.
            logger.exception(
                f"Store layer failed to get conversations for user {user_id}",
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"Failed to retrieve conversations for user {user_id}"
            ) from e
        except Exception as e:
            logger.exception(
                f"Unexpected error while getting conversations for user {user_id}",
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"An unexpected error occurred while retrieving conversations for user {user_id}"
            ) from e

    async def list_conversations_for_user(
        self,
        user_id: str,
        start: int,
        limit: int,
        sort: str = "updated_at",
        order: str = "desc",
    ) -> List[UserConversationListResponse]:
        """
        Returns a paginated list of conversations for a user, with agent display names resolved.
        All DB access is via the store layer; response assembly is handled here in the service.
        """
        try:
            conversations = await self.conversation_store.get_for_user(
                user_id=user_id,
                start=start,
                limit=limit,
                sort=sort,
                order=order,
            )

            # Resolve custom-agent IDs to display names in a single batch query.
            agent_ids = [c.agent_ids[0] for c in conversations if c.agent_ids]
            custom_agent_display_names: Dict[str, str] = {}
            if agent_ids:
                stmt = select(CustomAgentModel).where(
                    CustomAgentModel.id.in_(agent_ids)
                )
                result = await self.conversation_store.async_db.execute(stmt)
                agents = result.scalars().all()
                custom_agent_display_names = {agent.id: agent.role for agent in agents}

            response = []
            for conversation in conversations:
                projects = conversation.projects
                repo_name = projects[0].repo_name if projects else None
                branch_name = projects[0].branch_name if projects else None
                agent_id = conversation.agent_ids[0] if conversation.agent_ids else None
                display_agent_id = custom_agent_display_names.get(agent_id, agent_id)
                response.append(
                    UserConversationListResponse(
                        id=conversation.id,
                        title=conversation.title,
                        status=conversation.status,
                        project_ids=conversation.project_ids,
                        repository=repo_name,
                        branch=branch_name,
                        agent_id=display_agent_id,
                        created_at=conversation.created_at.isoformat(),
                        updated_at=conversation.updated_at.isoformat(),
                    )
                )
            return response
        except StoreError as e:
            logger.exception(
                f"Store layer failed to list conversations for user {user_id}",
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"Failed to retrieve conversations for user {user_id}"
            ) from e
        except Exception as e:
            logger.exception(
                f"Unexpected error listing conversations for user {user_id}",
                user_id=user_id,
            )
            raise ConversationServiceError(
                f"An unexpected error occurred while listing conversations for user {user_id}"
            ) from e
