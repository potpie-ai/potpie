import asyncio
import contextlib
import os
from collections.abc import Awaitable
from pathlib import Path

import aiohttp
import diskcache
from loguru import logger


class PotpieClient:
    def __init__(
        self,
        user_id: str,
        user_token: str,
        base_url: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.user_id = user_id
        self.user_token = user_token
        if base_url is None:
            base_url = os.getenv("POTPIE_BASE_URL", "http://localhost:8001")
        self.base_url = base_url.rstrip("/")
        self._external_session = session is not None
        self._session = session
        self.timeout = timeout or aiohttp.ClientTimeout(total=360)

    async def __aenter__(self) -> "PotpieClient":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._external_session and self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.user_token,
            "x-user-id": self.user_id,
        }

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        assert self._session is not None, "Client session is not initialized"

        async with self._session.request(
            method, url, headers=self.headers, **kwargs
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data
            else:
                raise Exception(
                    f"Failed to get response: {await resp.text()} {resp.status}",
                )

    async def post_parse(self, commit_id: str, repo_path: Path) -> str:
        payload = {"commit_id": commit_id, "repo_path": str(repo_path.absolute())}
        data = await self._request("POST", "/api/v2/parse", json=payload)
        return str(data["project_id"])

    async def get_parse_status(self, project_id: str) -> str:
        data = await self._request("GET", f"/api/v2/parsing-status/{project_id}")
        return str(data["status"]).lower()

    async def send_message(
        self, project_id: str, content: str, agent_id: str = "codebase_qna_agent"
    ) -> str:
        payload = {"content": content, "agent_id": agent_id}
        data = await self._request(
            "POST", f"/api/v2/project/{project_id}/message/", json=payload
        )
        return data["message"]

    async def get_available_projects(self):
        data = await self._request("GET", "/api/v2/projects/list")
        return data


class ReadinessCoordinator:
    """Coordinate readiness events for projects."""

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._in_progress: set[str] = set()
        self._lock = asyncio.Lock()

    async def ensure_event(self, project_id: str) -> asyncio.Event:
        # create-or-get in a single protected section
        async with self._lock:
            ev = self._events.get(project_id)
            if ev is None:
                ev = asyncio.Event()
                self._events[project_id] = ev
            return ev

    async def mark_ready(self, project_id: str) -> None:
        ev = await self.ensure_event(project_id)
        ev.set()

    async def wait_ready(self, project_id: str, timeout: float | None = None) -> None:
        ev = await self.ensure_event(project_id)
        if timeout is None:
            await ev.wait()
        else:
            await asyncio.wait_for(ev.wait(), timeout=timeout)

    async def try_acquire_enqueue_slot(self, project_id: str) -> bool:
        """Return True if the caller is allowed to enqueue the project (prevents duplicates)."""
        async with self._lock:
            if project_id in self._in_progress:
                return False
            self._in_progress.add(project_id)
            return True

    async def release_enqueue_slot(self, project_id: str) -> None:
        async with self._lock:
            self._in_progress.discard(project_id)


class PollingWorker:
    """Polls project statuses at a fixed interval (poll_interval_seconds)."""

    def __init__(
        self,
        client,
        coordinator,
        queue: asyncio.Queue,
        *,
        poll_interval_seconds: float = 5.0,
        max_attempts: int | None = None,
    ) -> None:
        self.client = client
        self.coordinator = coordinator
        self.queue = queue
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.max_attempts = None if max_attempts is None else int(max_attempts)
        self._stop_event = asyncio.Event()

    async def stop(self) -> None:
        """Request the worker to stop (non-blocking)."""
        self._stop_event.set()

    async def run(self) -> None:
        """Main worker loop. Run this in an asyncio.Task."""
        logger.info(
            "PollingWorker started",
            poll_interval=self.poll_interval_seconds,
        )
        try:
            while not self._stop_event.is_set():
                project_id = await self.queue.get()
                try:
                    await self._handle_project(project_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Unhandled error while handling project", project_id=project_id
                    )
                finally:
                    # Always mark the queue item as done
                    try:
                        self.queue.task_done()
                    except Exception:
                        pass
        except asyncio.CancelledError:
            logger.info("PollingWorker cancelled")
            raise
        finally:
            logger.info("PollingWorker stopped")

    async def _handle_project(self, project_id: str) -> None:
        """
        Polls status on a fixed interval until either:
          - status == "ready" -> mark ready + release slot
          - max_attempts reached -> release slot and log
          - stop requested or CancelledError -> release slot and exit
        """
        attempt = 0
        # ensure an Event exists so waiters can wait safely
        await self.coordinator.ensure_event(project_id)

        try:
            while not self._stop_event.is_set():
                attempt += 1
                try:
                    status = await self.client.get_parse_status(project_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if self.max_attempts is not None and attempt >= self.max_attempts:
                        logger.warning(
                            "Max attempts reached; releasing enqueue slot",
                            project_id=project_id,
                            attempt=attempt,
                        )
                        await self.coordinator.release_enqueue_slot(project_id)
                        return
                    await asyncio.sleep(self.poll_interval_seconds)
                    continue

                logger.debug(
                    "Polled {}: status={} (attempt={})", project_id, status, attempt
                )

                if status == "ready":
                    # notify waiters and free the in-progress marker
                    await self.coordinator.mark_ready(project_id)
                    await self.coordinator.release_enqueue_slot(project_id)
                    logger.info(
                        "Project is ready", project_id=project_id, attempt=attempt
                    )
                    return

                # not ready: check attempts and either stop or sleep fixed interval
                if self.max_attempts is not None and attempt >= self.max_attempts:
                    logger.warning(
                        "Giving up on {} after {} attempts (still status={}); releasing enqueue slot",
                        project_id,
                        attempt,
                        status,
                    )
                    await self.coordinator.release_enqueue_slot(project_id)
                    return

                # fixed-interval wait (no exponential backoff)
                await asyncio.sleep(self.poll_interval_seconds)

        finally:
            try:
                await self.coordinator.release_enqueue_slot(project_id)
            except Exception:
                logger.exception(
                    "Error releasing enqueue slot in finally", project_id=project_id
                )


async def _enqueue_project(
    coordinator: ReadinessCoordinator, queue: asyncio.Queue[str], project_id: str
):
    await coordinator.ensure_event(project_id)
    acquired = await coordinator.try_acquire_enqueue_slot(project_id)
    if acquired:
        await queue.put(project_id)
    else:
        logger.debug(
            "Project already in progress; not enqueuing again", project_id=project_id
        )


async def send_message(
    client: PotpieClient,
    coordinator: ReadinessCoordinator,
    project_id: str,
    content: str,
    agent_id: str = "codebase_qna_agent",
) -> str:
    await coordinator.wait_ready(project_id)
    return await client.send_message(project_id, content, agent_id)


async def get_all_st_answers(
    problems: list[dict[str, str]],
    repo_dict: dict[tuple[str, str], dict[tuple[str, int], Path]],
    task: str = "qa",
):
    project_cache = diskcache.Cache("project_cache")
    user_id = os.environ["defaultUsername"]
    user_token = os.environ["INTERNAL_ADMIN_SECRET"]
    coordinator = ReadinessCoordinator()
    queue: asyncio.Queue[str] = asyncio.Queue()

    async with PotpieClient(user_id=user_id, user_token=user_token) as client:
        worker = PollingWorker(client, coordinator, queue)
        worker_task = asyncio.create_task(worker.run())
        question_tasks: list[Awaitable[str]] = []
        try:
            for problem in problems:
                if task == "codegen":
                    # SWE-bench format: "django/django" -> "https://github.com/django/django"
                    repo_url = f"https://github.com/{problem['repo']}"
                    commit_id = problem["base_commit"]
                    problem_id = problem["instance_id"]
                else:
                    repo_url = problem["repo_url"]
                    commit_id = problem["commit_id"]
                    problem_id = problem["problem_id"]
                
                worktree_maps = repo_dict[(repo_url, commit_id)]
                # select a repo for project id caching
                # All worktrees are the same for now
                # NOTE: DON'T USE THIS FOR CODE GENERATION
                repo_path = sorted(worktree_maps.values())[0]
                cache_key = f"{repo_url}_{commit_id}"
                cached_project_id = project_cache.get(cache_key)
                project_id = None
                if cached_project_id is not None:
                    existing_projects = await client.get_available_projects()
                    existing_project_ids = {
                        project["id"] for project in existing_projects
                    }
                    if cached_project_id in existing_project_ids:
                        logger.info(
                            "Using cached project_id",
                            project_id=cached_project_id,
                            repo_url=repo_url,
                            commit_id=commit_id,
                        )
                        project_id = str(cached_project_id)

                if project_id is None:
                    project_id = await client.post_parse(commit_id, repo_path)
                    project_cache[cache_key] = project_id
                await _enqueue_project(coordinator, queue, project_id)

                i = 0
                while (problem_id, i) in worktree_maps:
                    question_tasks.append(
                        send_message(
                            client, coordinator, project_id, problem["question"]
                        )
                    )
                    i += 1

            answers = await asyncio.gather(*question_tasks)
            return answers
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task


async def get_all_codegen_answers(
    problems: list[dict[str, str]],
    repo_dict: dict[tuple[str, str], dict[tuple[str, int], Path]],
):
    """Get code generation answers from Potpie using the code_generation_agent.
    
    Important: Unlike QA, code generation requires SEPARATE project_ids for each batch
    because the agent modifies files. Each batch must operate on its own isolated worktree
    to prevent test contamination.
    
    Args:
        problems: List of problems with repo_url, commit_id, problem_id, and problem_statement
        repo_dict: Dictionary mapping (repo_url, commit_id) to {(problem_id, batch_idx): worktree_path}
    
    Returns:
        List of generated code/patch strings
    """
    project_cache = diskcache.Cache("project_cache_codegen")
    user_id = os.environ["defaultUsername"]
    user_token = os.environ["INTERNAL_ADMIN_SECRET"]
    coordinator = ReadinessCoordinator()
    queue: asyncio.Queue[str] = asyncio.Queue()

    async with PotpieClient(user_id=user_id, user_token=user_token) as client:
        worker = PollingWorker(client, coordinator, queue)
        worker_task = asyncio.create_task(worker.run())
        codegen_tasks: list[Awaitable[str]] = []
        try:
            for problem in problems:
                repo_url = f"https://github.com/{problem['repo']}"
                commit_id = problem["base_commit"]
                problem_id = problem["instance_id"]
                worktree_maps = repo_dict[(repo_url, commit_id)]

                # For code generation: Parse EACH worktree separately to get independent project_ids
                i = 0
                while (problem_id, i) in worktree_maps:
                    repo_path = worktree_maps[(problem_id, i)]
                    cache_key = f"{repo_url}_{commit_id}_{problem_id}_{i}"
                    cached_project_id = project_cache.get(cache_key)
                    project_id = None
                    
                    if cached_project_id is not None:
                        existing_projects = await client.get_available_projects()
                        existing_project_ids = {
                            project["id"] for project in existing_projects
                        }
                        if cached_project_id in existing_project_ids:
                            logger.info(
                                "Using cached project_id for worktree",
                                project_id=cached_project_id,
                                repo_url=repo_url,
                                commit_id=commit_id,
                                problem_id=problem_id,
                                batch=i,
                            )
                            project_id = str(cached_project_id)

                    if project_id is None:
                        project_id = await client.post_parse(commit_id, repo_path)
                        project_cache[cache_key] = project_id
                        logger.info(
                            "Parsed new worktree for code generation",
                            project_id=project_id,
                            repo_url=repo_url,
                            problem_id=problem_id,
                            batch=i,
                        )
                    
                    await _enqueue_project(coordinator, queue, project_id)
                    
                    # Each task uses its own project_id (pointing to its own worktree)
                    codegen_tasks.append(
                        send_message(
                            client,
                            coordinator,
                            project_id,
                            problem["problem_statement"],
                            agent_id="code_generation_agent",
                        )
                    )
                    i += 1

            answers = await asyncio.gather(*codegen_tasks)
            return answers
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task


if __name__ == "__main__":
    from dotenv import load_dotenv

    _ = load_dotenv()
    asyncio.run(get_all_st_answers(None, None))
