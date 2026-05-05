from unittest.mock import AsyncMock, MagicMock

import pytest

from app.modules.parsing.graph_construction.parsing_service import ParsingService
import app.modules.parsing.graph_construction.parsing_service as parsing_service_module
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.projects.projects_schema import ProjectStatusEnum


pytestmark = pytest.mark.unit


class DummyInferenceService:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def log_graph_stats(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_schedule_colgrep_index_build_dispatches_celery_task(monkeypatch):
    monkeypatch.setattr(
        parsing_service_module, "InferenceService", DummyInferenceService
    )

    db = MagicMock()
    service = ParsingService(db=db, user_id="defaultuser", raise_library_exceptions=False)
    monkeypatch.setattr(service, "_should_run_colgrep_index_locally", lambda: False)
    from app.celery.tasks import parsing_tasks as parsing_tasks_module

    delay_mock = MagicMock(return_value=MagicMock(id="colgrep-task-1"))
    monkeypatch.setattr(parsing_tasks_module.process_colgrep_index, "delay", delay_mock)

    service._schedule_colgrep_index_build("/tmp/worktree")

    delay_mock.assert_called_once_with(
        "/tmp/worktree",
        str(service.repo_manager.repos_base_path),
    )


@pytest.mark.asyncio
async def test_schedule_colgrep_index_build_falls_back_to_local_thread(monkeypatch):
    monkeypatch.setattr(
        parsing_service_module, "InferenceService", DummyInferenceService
    )

    db = MagicMock()
    service = ParsingService(db=db, user_id="defaultuser", raise_library_exceptions=False)
    monkeypatch.setattr(service, "_should_run_colgrep_index_locally", lambda: False)
    from app.celery.tasks import parsing_tasks as parsing_tasks_module

    def raise_delay(*args, **kwargs):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(parsing_tasks_module.process_colgrep_index, "delay", raise_delay)

    captured: dict[str, object] = {}

    def fake_build(repo_root, repos_base_path):
        captured["build_call"] = (repo_root, repos_base_path)

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            captured["thread_name"] = name
            captured["thread_daemon"] = daemon
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(parsing_service_module, "build_colgrep_index", fake_build)
    monkeypatch.setattr(parsing_service_module.threading, "Thread", FakeThread)

    service._schedule_colgrep_index_build("/tmp/worktree")

    assert captured["build_call"] == (
        "/tmp/worktree",
        service.repo_manager.repos_base_path,
    )
    assert captured["thread_name"] == "colgrep-init-worktree"
    assert captured["thread_daemon"] is True


@pytest.mark.asyncio
async def test_schedule_colgrep_index_build_uses_local_thread_for_local_runtime(monkeypatch):
    monkeypatch.setattr(
        parsing_service_module, "InferenceService", DummyInferenceService
    )

    db = MagicMock()
    service = ParsingService(db=db, user_id="defaultuser", raise_library_exceptions=False)
    monkeypatch.setattr(service, "_should_run_colgrep_index_locally", lambda: True)
    from app.celery.tasks import parsing_tasks as parsing_tasks_module

    delay_mock = MagicMock()
    monkeypatch.setattr(parsing_tasks_module.process_colgrep_index, "delay", delay_mock)

    captured: dict[str, object] = {}

    def fake_build(repo_root, repos_base_path):
        captured["build_call"] = (repo_root, repos_base_path)

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            captured["thread_name"] = name
            captured["thread_daemon"] = daemon
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(parsing_service_module, "build_colgrep_index", fake_build)
    monkeypatch.setattr(parsing_service_module.threading, "Thread", FakeThread)

    service._schedule_colgrep_index_build("/tmp/worktree")

    delay_mock.assert_not_called()
    assert captured["build_call"] == (
        "/tmp/worktree",
        service.repo_manager.repos_base_path,
    )
    assert captured["thread_name"] == "colgrep-init-worktree"
    assert captured["thread_daemon"] is True


@pytest.mark.asyncio
async def test_short_circuit_marks_ready_and_returns_early(monkeypatch):
    monkeypatch.setenv("PARSING_SHORT_CIRCUIT_REPOMANAGER", "true")
    monkeypatch.setattr(
        "app.modules.parsing.graph_construction.parsing_service.config_provider.get_is_development_mode",
        lambda: False,
    )
    monkeypatch.setattr(
        parsing_service_module, "InferenceService", DummyInferenceService
    )

    db = MagicMock()
    service = ParsingService(db=db, user_id="defaultuser", raise_library_exceptions=False)

    # Replace heavy collaborators with mocks.
    service.project_service = MagicMock()
    service.project_service.get_project_from_db_by_id = AsyncMock(return_value=None)
    service.project_service.update_project_status = AsyncMock()

    service.parse_helper = MagicMock()
    service.parse_helper.clone_or_copy_repository = AsyncMock(
        return_value=(None, None, None, "/tmp/worktree")
    )
    service.parse_helper.setup_project_directory = AsyncMock(
        return_value=("/tmp/worktree", "ignored_project_id_return")
    )
    service._schedule_colgrep_index_build = MagicMock()

    req = ParsingRequest(
        repo_name="root/potpie",
        branch_name="main",
        repo_path=None,
        commit_id=None,
    )

    # The call should return before any heavy analysis/language detection is invoked.
    resp = await service.parse_directory(
        repo_details=req,
        user_id="defaultuser",
        user_email="",
        project_id="proj-1",
        cleanup_graph=False,
    )

    assert resp["id"] == "proj-1"
    assert resp["message"].startswith("Project marked READY")
    service.project_service.update_project_status.assert_any_call(
        "proj-1", ProjectStatusEnum.READY
    )
    service._schedule_colgrep_index_build.assert_not_called()
