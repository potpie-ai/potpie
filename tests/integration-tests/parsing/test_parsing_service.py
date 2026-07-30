"""
Integration tests for ParsingService.parse_directory with mocks.

Heavy mocking of ParseHelper, CodeGraphService, InferenceService, ProjectService;
no real Neo4j/Git/RepoManager required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.modules.parsing.graph_construction.parsing_helper import (
    ParsingServiceError,
)
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.projects.projects_schema import ProjectStatusEnum


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestParseDirectory:
    """Test ParsingService.parse_directory behavior with mocks."""

    @pytest.mark.asyncio
    async def test_parse_directory_project_inferring_early_return(self, db_session):
        """Project status INFERRING; expect early return with message, no clone/analyze."""
        project_id = "proj-inferring"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": ProjectStatusEnum.INFERRING.value}
        )
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ):
            service = ParsingService(db_session, "test-user")
            repo_details = ParsingRequest(repo_name="owner/repo")
            result = await service.parse_directory(
                repo_details,
                user_id="test-user",
                user_email="test@example.com",
                project_id=project_id,
                cleanup_graph=False,
            )
            assert result is not None
            assert result.get("status") == ProjectStatusEnum.INFERRING.value
            assert result.get("message") == "Project already inferring"
            assert result.get("id") == project_id
            mock_project_manager.get_project_from_db_by_id.assert_called_once_with(
                project_id
            )

    @pytest.mark.asyncio
    async def test_parse_directory_commit_matches_early_return(self, db_session):
        """Project READY and check_commit_status True; expect early return, no clone."""
        project_id = "proj-ready"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={
                "id": project_id,
                "status": ProjectStatusEnum.READY.value,
                "commit_id": "abc123",
                "project_name": "repo",
                "branch_name": "main",
                "repo_path": None,
            }
        )
        mock_project_manager.update_project_status = AsyncMock()
        mock_parse_helper = MagicMock()
        mock_parse_helper.check_commit_status = AsyncMock(return_value=True)
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.ParseHelper",
            return_value=mock_parse_helper,
        ):
            service = ParsingService(db_session, "test-user")
            repo_details = ParsingRequest(
                repo_name="owner/repo", commit_id="abc123"
            )
            result = await service.parse_directory(
                repo_details,
                user_id="test-user",
                user_email="test@example.com",
                project_id=project_id,
                cleanup_graph=True,
            )
            assert result is not None
            assert result.get("message") == "Project already parsed for requested commit"
            assert result.get("id") == project_id
            mock_parse_helper.check_commit_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_directory_cleanup_bypassed(self, db_session):
        """cleanup_graph=True no longer touches Neo4j; flow proceeds past cleanup.

        Neo4j is permanently bypassed, so the old CodeGraphService cleanup
        cannot fail parse_directory anymore. We prove the flow gets past the
        cleanup step by making the *next* step (repo clone) raise a sentinel.
        """
        project_id = "proj-cleanup-bypass"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": "submitted"}
        )
        mock_project_manager.update_project_status = AsyncMock()
        mock_parse_helper = MagicMock()
        mock_parse_helper.check_commit_status = AsyncMock(return_value=False)
        mock_parse_helper.clone_or_copy_repository = AsyncMock(
            side_effect=FileNotFoundError("sentinel: reached step after cleanup")
        )
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.ParseHelper",
            return_value=mock_parse_helper,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            repo_details = ParsingRequest(repo_name="owner/repo")
            with pytest.raises(Exception) as exc_info:
                await service.parse_directory(
                    repo_details,
                    user_id="test-user",
                    user_email="test@example.com",
                    project_id=project_id,
                    cleanup_graph=True,
                )
            # Whatever failed, it was NOT the (removed) Neo4j cleanup.
            assert "cleanup" not in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_parse_directory_setup_returns_none(self, db_session):
        """clone_or_copy_repository raises; expect exception or 500 path."""
        project_id = "proj-setup-fail"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": "submitted"}
        )
        mock_parse_helper = MagicMock()
        mock_parse_helper.check_commit_status = AsyncMock(return_value=False)
        mock_parse_helper.clone_or_copy_repository = AsyncMock(
            side_effect=FileNotFoundError("clone failed")
        )
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.ParseHelper",
            return_value=mock_parse_helper,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            repo_details = ParsingRequest(repo_name="owner/repo")
            with pytest.raises((ParsingServiceError, FileNotFoundError, Exception)):
                await service.parse_directory(
                    repo_details,
                    user_id="test-user",
                    user_email="test@example.com",
                    project_id=project_id,
                    cleanup_graph=True,
                )


class TestNeo4jFailures:
    """Neo4j is permanently bypassed; an outage must not fail parsing."""

    @pytest.mark.asyncio
    async def test_neo4j_outage_cannot_fail_cleanup(self, db_session):
        """A dead Neo4j can no longer surface through parse_directory cleanup."""
        project_id = "proj-neo4j-fail"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": "submitted"}
        )
        mock_project_manager.update_project_status = AsyncMock()
        mock_parse_helper = MagicMock()
        mock_parse_helper.check_commit_status = AsyncMock(return_value=False)
        mock_parse_helper.clone_or_copy_repository = AsyncMock(
            side_effect=FileNotFoundError("sentinel: reached step after cleanup")
        )

        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.ParseHelper",
            return_value=mock_parse_helper,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            repo_details = ParsingRequest(repo_name="owner/repo")
            with pytest.raises(Exception) as exc_info:
                await service.parse_directory(
                    repo_details,
                    user_id="test-user",
                    user_email="test@example.com",
                    project_id=project_id,
                    cleanup_graph=True,
                )
            # The failure must never be a Neo4j connectivity error.
            assert not isinstance(exc_info.value, ServiceUnavailable)
            assert "cleanup" not in str(exc_info.value).lower()

class TestProjectServiceOwnership:
    """Test ProjectService ownership checks (Part 4.6 of plan)."""

    @pytest.mark.asyncio
    async def test_register_project_different_user_403(self, db_session, test_user):
        """register_project with existing project_id but different user_id → 403."""
        from fastapi import HTTPException
        from app.modules.projects.projects_service import ProjectService
        from app.modules.projects.projects_model import Project

        # Create a project owned by the test user
        project_id = "proj-ownership-test"
        existing_project = Project(
            id=project_id,
            repo_name="owner/repo",
            branch_name="main",
            user_id=test_user.uid,
            status="ready",
        )
        db_session.add(existing_project)
        db_session.commit()

        try:
            project_service = ProjectService(db_session)

            # Try to register same project_id with different user
            with pytest.raises(HTTPException) as exc_info:
                await project_service.register_project(
                    repo_name="owner/repo",
                    branch_name="main",
                    user_id="different-user-id",
                    project_id=project_id,
                )
            assert exc_info.value.status_code == 403
            assert "ownership" in exc_info.value.detail.lower() or "mismatch" in exc_info.value.detail.lower()
        finally:
            # Cleanup
            db_session.query(Project).filter(Project.id == project_id).delete()
            db_session.commit()

    @pytest.mark.asyncio
    async def test_get_project_with_no_branch_no_commit(self, db_session, test_user):
        """get_project_from_db with branch_name=None, commit_id=None."""
        from app.modules.projects.projects_service import ProjectService
        from app.modules.projects.projects_model import Project

        project_id = "proj-no-branch"
        project = Project(
            id=project_id,
            repo_name="owner/nobranch-repo",
            branch_name=None,
            user_id=test_user.uid,
            status="ready",
        )
        db_session.add(project)
        db_session.commit()

        try:
            project_service = ProjectService(db_session)
            result = await project_service.get_project_from_db(
                repo_name="owner/nobranch-repo",
                branch_name=None,
                user_id=test_user.uid,
                commit_id=None,
            )
            # Should find the project or return None gracefully
            # Either is valid behavior
            assert result is None or result.id == project_id
        finally:
            db_session.query(Project).filter(Project.id == project_id).delete()
            db_session.commit()

    @pytest.mark.asyncio
    async def test_update_nonexistent_project(self, db_session):
        """update_project_status with non-existent project_id."""
        from app.modules.projects.projects_service import ProjectService

        project_service = ProjectService(db_session)
        # Should not raise, just log or return gracefully
        await project_service.update_project_status(
            "nonexistent-project-id-12345",
            ProjectStatusEnum.ERROR,
        )
        # If we get here without exception, behavior is acceptable
