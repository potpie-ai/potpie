import asyncio
import json
import os
from asyncio import create_task
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict

from dotenv import load_dotenv
from fastapi import HTTPException
import redis
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.core.config_provider import ConfigProvider
from app.core.database import SessionLocal
from app.celery.tasks.parsing_tasks import process_parsing
from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    Visibility,
)
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import (
    ParsingRequest,
    ParsingStatusRequest,
)
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
from app.modules.parsing.utils.repo_name_normalizer import normalize_repo_name
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.utils.email_helper import EmailHelper
from app.modules.utils.logger import setup_logger
from app.modules.utils.posthog_helper import PostHogClient

logger = setup_logger(__name__)

load_dotenv(override=True)


class ParsingController:
    INTERNAL_SERVER_ERROR = "Internal server error"
    STREAM_ERROR_MESSAGE = "Unable to stream parsing status"
    _background_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _format_sse_event(event_name: str, payload: Dict[str, Any]) -> str:
        """Render one SSE frame with a named event and JSON payload."""
        return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

    @staticmethod
    def _utc_timestamp() -> str:
        """Return current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _status_payload_with_metadata(
        project_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attach stream metadata fields to a status payload."""
        return {
            **payload,
            "project_id": project_id,
            "timestamp": ParsingController._utc_timestamp(),
        }

    @staticmethod
    def _error_payload(
        project_id: str,
        detail: str,
        status_code: int | None = None,
    ) -> Dict[str, Any]:
        """Build a standardized SSE error payload."""
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "detail": detail,
            "timestamp": ParsingController._utc_timestamp(),
        }
        if status_code is not None:
            payload["status_code"] = status_code
        return payload

    @staticmethod
    def _is_terminal_status(status: str | None) -> bool:
        """Return True when parsing status is terminal."""
        return status in {ProjectStatusEnum.READY.value, ProjectStatusEnum.ERROR.value}

    @staticmethod
    def _track_background_task(task: asyncio.Task, task_name: str) -> None:
        """Retain and monitor fire-and-forget tasks to prevent premature GC."""
        ParsingController._background_tasks.add(task)

        def _on_task_done(done_task: asyncio.Task) -> None:
            ParsingController._background_tasks.discard(done_task)
            if done_task.cancelled():
                return
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.exception("Background task '%s' failed", task_name, exc_info=exc)

        task.add_done_callback(_on_task_done)

    @staticmethod
    async def _read_status_events(
        redis_client: redis.Redis,
        stream_key: str,
        last_id: str,
    ) -> Any:
        """Block on Redis stream for the next status event batch."""
        return await asyncio.to_thread(
            lambda last_id=last_id: redis_client.xread(
                {stream_key: last_id}, block=5000, count=1
            )
        )

    @staticmethod
    async def _fetch_stream_status_payload(
        project_id: str,
        user: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch parsing status and enrich it for SSE delivery."""
        with SessionLocal() as status_db:
            payload = await ParsingController.fetch_parsing_status(
                project_id, status_db, user
            )
        return ParsingController._status_payload_with_metadata(project_id, payload)

    @staticmethod
    def _initial_stream_cursor(redis_client: redis.Redis, stream_key: str) -> str:
        """Return a cursor anchored to current stream tail to avoid missing new events."""
        try:
            latest_events = redis_client.xrevrange(stream_key, count=1)
            if latest_events:
                latest_id, _ = latest_events[0]
                return (
                    latest_id.decode()
                    if isinstance(latest_id, bytes)
                    else str(latest_id)
                )
        except Exception:
            logger.exception("Failed to read initial parsing stream cursor")
        return "0-0"

    @staticmethod
    async def _stream_status_updates(
        project_id: str,
        user: Dict[str, Any],
        request: Any,
        redis_client: redis.Redis,
        stream_key: str,
        started_at: float,
        last_emitted_at: float,
        last_id: str,
        heartbeat_interval_seconds: float,
        max_stream_duration_seconds: float,
    ) -> AsyncGenerator[str, None]:
        """Emit incremental SSE updates from Redis stream events until completion."""
        loop = asyncio.get_running_loop()

        while True:
            if await request.is_disconnected():
                return

            if loop.time() - started_at >= max_stream_duration_seconds:
                timeout_payload = ParsingController._error_payload(
                    project_id, "Parsing status stream timed out"
                )
                yield ParsingController._format_sse_event("error", timeout_payload)
                return

            try:
                events = await ParsingController._read_status_events(
                    redis_client, stream_key, last_id
                )
            except Exception as e:
                logger.exception("Error reading parsing status events")
                error_payload = ParsingController._error_payload(
                    project_id, ParsingController.STREAM_ERROR_MESSAGE
                )
                yield ParsingController._format_sse_event("error", error_payload)
                return

            if not events:
                now = loop.time()
                if now - last_emitted_at >= heartbeat_interval_seconds:
                    heartbeat_payload = {
                        "project_id": project_id,
                        "timestamp": ParsingController._utc_timestamp(),
                    }
                    yield ParsingController._format_sse_event(
                        "heartbeat", heartbeat_payload
                    )
                    last_emitted_at = now
                continue

            for _, stream_events in events:
                for event_id, _ in stream_events:
                    last_id = event_id

                    try:
                        status_payload = (
                            await ParsingController._fetch_stream_status_payload(
                                project_id, user
                            )
                        )
                    except HTTPException as e:
                        client_detail = (
                            str(e.detail)
                            if e.status_code < 500
                            else ParsingController.INTERNAL_SERVER_ERROR
                        )
                        error_payload = ParsingController._error_payload(
                            project_id, client_detail, e.status_code
                        )
                        yield ParsingController._format_sse_event("error", error_payload)
                        return
                    except Exception as e:
                        logger.exception("Error fetching parsing status while streaming")
                        error_payload = ParsingController._error_payload(
                            project_id,
                            ParsingController.STREAM_ERROR_MESSAGE,
                        )
                        yield ParsingController._format_sse_event("error", error_payload)
                        return

                    yield ParsingController._format_sse_event("status", status_payload)

                    if ParsingController._is_terminal_status(status_payload.get("status")):
                        yield ParsingController._format_sse_event("complete", status_payload)
                        return

                    last_emitted_at = loop.time()

    @staticmethod
    async def stream_parsing_status(
        project_id: str,
        db: Session,
        user: Dict[str, Any],
        request: Any,
        pre_fetched_status: Dict[str, Any] | None = None,
        heartbeat_interval_seconds: float = 15.0,
        max_stream_duration_seconds: float = 600.0,
    ):
        """Stream parsing status over SSE using Redis-triggered updates."""
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        last_emitted_at = started_at
        redis_client = redis.from_url(
            ConfigProvider().get_redis_url(),
            socket_connect_timeout=10,
            socket_timeout=30,
            decode_responses=False,
        )
        stream_key = f"parsing:stream:{project_id}"
        last_id = ParsingController._initial_stream_cursor(redis_client, stream_key)

        try:
            try:
                if pre_fetched_status is not None:
                    initial_payload = ParsingController._status_payload_with_metadata(
                        project_id, pre_fetched_status
                    )
                else:
                    initial_payload = (
                        await ParsingController._fetch_stream_status_payload(
                            project_id, user
                        )
                    )
                yield ParsingController._format_sse_event("status", initial_payload)
                last_emitted_at = loop.time()

                if ParsingController._is_terminal_status(initial_payload.get("status")):
                    yield ParsingController._format_sse_event("complete", initial_payload)
                    return
            except HTTPException as e:
                client_detail = (
                    str(e.detail)
                    if e.status_code < 500
                    else ParsingController.INTERNAL_SERVER_ERROR
                )
                error_payload = ParsingController._error_payload(
                    project_id, client_detail, e.status_code
                )
                yield ParsingController._format_sse_event("error", error_payload)
                return
            except Exception as e:
                logger.error(f"Error in stream_parsing_status: {str(e)}")
                error_payload = ParsingController._error_payload(
                    project_id, ParsingController.INTERNAL_SERVER_ERROR
                )
                yield ParsingController._format_sse_event("error", error_payload)
                return
            async for event in ParsingController._stream_status_updates(
                project_id,
                user,
                request,
                redis_client,
                stream_key,
                started_at,
                last_emitted_at,
                last_id,
                heartbeat_interval_seconds,
                max_stream_duration_seconds,
            ):
                yield event
        finally:
            try:
                await asyncio.to_thread(redis_client.close)
            except Exception:
                logger.exception("Failed to close redis client for parsing status stream")

    @staticmethod
    @validate_parsing_input
    async def parse_directory(
        repo_details: ParsingRequest, db: Session, user: Dict[str, Any]
    ):
        if "email" not in user:
            user_email = None
        else:
            user_email = user["email"]

        user_id = user["user_id"]
        project_manager = ProjectService(db)
        parse_helper = ParseHelper(db)
        parsing_service = ParsingService(db, user_id)

        # Auto-detect if repo_name is actually a filesystem path
        if repo_details.repo_name and not repo_details.repo_path:
            is_path = (
                os.path.isabs(repo_details.repo_name)
                or repo_details.repo_name.startswith(("~", "./", "../"))
                or os.path.isdir(os.path.expanduser(repo_details.repo_name))
            )
            if is_path:
                # Move from repo_name to repo_path
                repo_details.repo_path = repo_details.repo_name
                repo_details.repo_name = repo_details.repo_path.split("/")[-1]
                logger.info(
                    f"Auto-detected filesystem path: repo_path={repo_details.repo_path}, repo_name={repo_details.repo_name}"
                )

        if config_provider.get_is_development_mode():
            # In dev mode: if both repo_path and repo_name are provided, prioritize repo_path (local)
            if repo_details.repo_path and repo_details.repo_name:
                repo_details.repo_name = None
            # Otherwise keep whichever one is provided as-is
        else:
            # In non-dev mode: if repo_name is None but repo_path exists, extract repo_name from repo_path
            if not repo_details.repo_name and repo_details.repo_path:
                repo_details.repo_name = repo_details.repo_path.split("/")[-1]

        # For later use in the code
        repo_name = repo_details.repo_name or (
            repo_details.repo_path.split("/")[-1] if repo_details.repo_path else None
        )
        repo_path = repo_details.repo_path
        if repo_path:
            if os.getenv("isDevelopmentMode") != "enabled":
                raise HTTPException(
                    status_code=400,
                    detail="Parsing local repositories is only supported in development mode",
                )
            else:
                new_project_id = str(uuid7())
                return await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )
        try:
            # Normalize repository name for consistent database lookups
            normalized_repo_name = normalize_repo_name(repo_name)
            logger.debug(
                f"Original repo_name: {repo_name}, Normalized: {normalized_repo_name}"
            )

            project = await project_manager.get_project_from_db(
                normalized_repo_name,
                repo_details.branch_name,
                user_id,
                repo_path=repo_details.repo_path,
                commit_id=repo_details.commit_id,
            )
            demo_repos = [
                "calcom/cal.com",
                "langchain-ai/langchain",
                "electron/electron",
                "openclaw/openclaw",
                "pydantic/pydantic-ai",
            ]
            if not project and repo_details.repo_name in demo_repos:
                existing_project = await project_manager.get_global_project_from_db(
                    normalized_repo_name,
                    repo_details.branch_name,
                    repo_details.commit_id,
                )

                new_project_id = str(uuid7())

                if existing_project:
                    await project_manager.duplicate_project(
                        repo_name,
                        repo_details.branch_name,
                        user_id,
                        new_project_id,
                        existing_project.properties,
                        existing_project.commit_id,
                    )
                    await project_manager.update_project_status(
                        new_project_id, ProjectStatusEnum.SUBMITTED
                    )

                    old_project_id = await project_manager.get_demo_project_id(
                        repo_name
                    )

                    task = asyncio.create_task(
                        CodeProviderService(db).get_project_structure_async(
                            new_project_id
                        )
                    )

                    def _on_structure_done(t: asyncio.Task) -> None:
                        if t.cancelled():
                            return
                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            return
                        if exc is not None:
                            logger.exception(
                                "Failed to get project structure", exc_info=exc
                            )

                    task.add_done_callback(_on_structure_done)
                    # Duplicate the graph under the new repo ID
                    await parsing_service.duplicate_graph(
                        old_project_id, new_project_id
                    )

                    # Update the project status to READY after copying
                    await project_manager.update_project_status(
                        new_project_id, ProjectStatusEnum.READY
                    )
                    email_task = create_task(
                        EmailHelper().send_email(
                            user_email, repo_name, repo_details.branch_name
                        )
                    )

                    def _on_email_done(t: asyncio.Task) -> None:
                        if t.cancelled():
                            return
                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            return
                        if exc is not None:
                            logger.exception("Failed to send email", exc_info=exc)

                    email_task.add_done_callback(_on_email_done)

                    return {
                        "project_id": new_project_id,
                        "status": ProjectStatusEnum.READY.value,
                    }
                else:
                    return await ParsingController.handle_new_project(
                        repo_details,
                        user_id,
                        user_email,
                        new_project_id,
                        project_manager,
                        db,
                    )

            # Handle existing projects (including previously duplicated demo projects)
            if project:
                project_id = project.id

                # If project is already inferring, return current state (don't re-submit parse)
                if project.status == ProjectStatusEnum.INFERRING.value:
                    logger.info(
                        f"Project {project_id} already in inferring state. Returning current state."
                    )
                    return {"project_id": project_id, "status": project.status}

                # Check if this project is already parsed for the requested commit
                # Only check commit status if commit_id is provided
                if repo_details.commit_id:
                    is_latest = await parse_helper.check_commit_status(
                        project_id, requested_commit_id=repo_details.commit_id
                    )
                else:
                    # If no commit_id provided, check if project is READY (assume it's for the branch)
                    is_latest = project.status == ProjectStatusEnum.READY.value

                # If project exists with this commit_id and is READY, return it immediately
                if is_latest and project.status == ProjectStatusEnum.READY.value:
                    logger.info(
                        f"Project {project_id} already exists and is READY for commit {repo_details.commit_id or 'branch'}. "
                        "Returning existing project."
                    )
                    # Ensure worktree exists in repo manager when enabled
                    if os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true":
                        repo_name = str(project.repo_name) if project.repo_name is not None else None
                        branch = str(project.branch_name) if project.branch_name is not None else None
                        commit_id_val = str(project.commit_id) if project.commit_id is not None else None
                        repo_path = str(project.repo_path) if project.repo_path is not None else None
                        if repo_name and not repo_path:
                            ref = commit_id_val if commit_id_val else branch
                            if ref:
                                from app.modules.code_provider.github.github_service import GithubService  # noqa: PLC0415
                                _repo_manager = RepoManager()
                                try:
                                    _auth_token = GithubService(db).get_github_oauth_token(user_id)
                                except Exception:
                                    _auth_token = None

                                async def _ensure_worktree_bg(
                                    _rm=_repo_manager,
                                    _rn=repo_name,
                                    _ref=ref,
                                    _at=_auth_token,
                                    _ic=bool(commit_id_val),
                                    _uid=user_id,
                                ):
                                    try:
                                        await asyncio.get_event_loop().run_in_executor(
                                            None,
                                            lambda: _rm.prepare_for_parsing(
                                                _rn, _ref, auth_token=_at, is_commit=_ic, user_id=_uid
                                            ),
                                        )
                                        logger.info(
                                            "Background worktree ensured for READY project %s (%s@%s)",
                                            project_id,
                                            _rn,
                                            _ref,
                                        )
                                    except Exception:
                                        logger.warning(
                                            "Background worktree failed for project %s",
                                            project_id,
                                            exc_info=True,
                                        )

                                worktree_task = asyncio.create_task(_ensure_worktree_bg())
                                ParsingController._track_background_task(
                                    worktree_task, "ensure_worktree_bg"
                                )
                    return {"project_id": project_id, "status": project.status}

                # If project exists but commit doesn't match or status is not READY, reparse
                cleanup_graph = True
                logger.info(
                    "Submitting parsing task for existing project.",
                    project_id=project_id,
                    is_latest=is_latest,
                    status=project.status,
                )
                try:
                    task = process_parsing.delay(
                        repo_details.model_dump(),
                        user_id,
                        user_email,
                        project_id,
                        cleanup_graph,
                    )
                    logger.info(
                        "Parsing task submitted to Celery",
                        task_id=task.id,
                        project_id=project_id,
                    )
                except Exception as e:
                    logger.exception(
                        "Failed to submit parsing task to Celery",
                        project_id=project_id,
                        error=str(e),
                    )
                    raise

                await project_manager.update_project_status(
                    project_id, ProjectStatusEnum.SUBMITTED
                )
                PostHogClient().send_event(
                    user_id,
                    "parsed_repo_event",
                    {
                        "repo_name": repo_details.repo_name,
                        "branch": repo_details.branch_name,
                        "commit_id": repo_details.commit_id,
                        "project_id": project_id,
                    },
                )
                return {
                    "project_id": project_id,
                    "status": ProjectStatusEnum.SUBMITTED.value,
                }
            else:
                # Handle new non-demo projects
                new_project_id = str(uuid7())
                return await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )

        except Exception as e:
            logger.error(f"Error in parse_directory: {e}")
            raise HTTPException(
                status_code=500, detail=ParsingController.INTERNAL_SERVER_ERROR
            )

    @staticmethod
    async def handle_new_project(
        repo_details: ParsingRequest,
        user_id: str,
        user_email: str | None,
        new_project_id: str,
        project_manager: ProjectService,
        db: Session,
    ):
        response = {
            "project_id": new_project_id,
            "status": ProjectStatusEnum.SUBMITTED.value,
        }

        logger.info(f"Submitting parsing task for new project {new_project_id}")
        repo_name = repo_details.repo_name or repo_details.repo_path.split("/")[-1]
        await project_manager.register_project(
            repo_name,
            repo_details.branch_name,
            user_id,
            new_project_id,
            repo_details.commit_id,
            repo_details.repo_path,
        )
        # asyncio.create_task(
        #     CodeProviderService(db).get_project_structure_async(new_project_id)
        # )
        if not user_email:
            user_email = None

        process_parsing.delay(
            repo_details.model_dump(),
            user_id,
            user_email,
            new_project_id,
            False,
        )
        PostHogClient().send_event(
            user_id,
            "repo_parsed_event",
            {
                "repo_name": repo_details.repo_name,
                "branch": repo_details.branch_name,
                "commit_id": repo_details.commit_id,
                "project_id": new_project_id,
            },
        )
        return response

    @staticmethod
    async def fetch_parsing_status(
        project_id: str, db: Session, user: Dict[str, Any]
    ):
        try:
            project_query = (
                select(Project.status)
                .join(
                    Conversation, Conversation.project_ids.any(Project.id), isouter=True
                )
                .where(
                    Project.id == project_id,
                    or_(
                        Project.user_id == user["user_id"],
                        Conversation.visibility == Visibility.PUBLIC,
                        Conversation.shared_with_emails.any(user.get("email", "")),
                    ),
                )
                .limit(1)
            )

            result = db.execute(project_query)
            project_status = result.scalars().first()

            if not project_status:
                raise HTTPException(
                    status_code=404, detail="Project not found or access denied"
                )
            parse_helper = ParseHelper(db)
            is_latest = await parse_helper.check_commit_status(project_id)

            return {"status": project_status, "latest": is_latest}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in fetch_parsing_status: {str(e)}")
            raise HTTPException(
                status_code=500, detail=ParsingController.INTERNAL_SERVER_ERROR
            )

    @staticmethod
    async def fetch_parsing_status_by_repo(
        request: ParsingStatusRequest, db: AsyncSession, user: Dict[str, Any]
    ):
        try:
            user_id = user["user_id"]
            project_manager = ProjectService(db)

            # Use ProjectService to find project by repo_name and commit_id/branch_name
            normalized_repo_name = normalize_repo_name(request.repo_name)
            project = await project_manager.get_project_from_db(
                normalized_repo_name,
                request.branch_name,
                user_id,
                repo_path=None,
                commit_id=request.commit_id,
            )

            if not project:
                raise HTTPException(
                    status_code=404,
                    detail="Project not found for the given repo_name and commit_id/branch_name",
                )

            parse_helper = ParseHelper(db)
            is_latest = await parse_helper.check_commit_status(project.id)

            return {
                "project_id": project.id,
                "repo_name": project.repo_name,
                "status": project.status,
                "latest": is_latest,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in fetch_parsing_status_by_repo: {str(e)}")
            raise HTTPException(
                status_code=500, detail=ParsingController.INTERNAL_SERVER_ERROR
            )
