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

