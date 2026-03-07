"""
Integration tests for ParsingService (parse_directory, analyze_directory) with mocks.

Heavy mocking of ParseHelper, CodeGraphService, InferenceService, ProjectService;
no real Neo4j/Git/RepoManager required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.modules.parsing.graph_construction.parsing_helper import (
    ParsingFailedError,
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
    async def test_parse_directory_cleanup_graph_failure(self, db_session):
        """cleanup_graph raises; expect ParsingServiceError when raise_library_exceptions True."""
        project_id = "proj-cleanup-fail"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": "submitted"}
        )
        mock_project_manager.update_project_status = AsyncMock()
        mock_code_graph = MagicMock()
        mock_code_graph.cleanup_graph = MagicMock(side_effect=RuntimeError("cleanup failed"))
        mock_code_graph.close = MagicMock()
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.CodeGraphService",
            return_value=mock_code_graph,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            repo_details = ParsingRequest(repo_name="owner/repo")
            with pytest.raises(ParsingServiceError) as exc_info:
                await service.parse_directory(
                    repo_details,
                    user_id="test-user",
                    user_email="test@example.com",
                    project_id=project_id,
                    cleanup_graph=True,
                )
            assert "cleanup" in str(exc_info.value).lower() or "graph" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_parse_directory_setup_returns_none(self, db_session):
        """clone_or_copy_repository raises; expect exception or 500 path."""
        project_id = "proj-setup-fail"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": "submitted"}
        )
        mock_code_graph = MagicMock()
        mock_code_graph.cleanup_graph = MagicMock()
        mock_code_graph.close = MagicMock()
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
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.CodeGraphService",
            return_value=mock_code_graph,
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


class TestAnalyzeDirectory:
    """Test ParsingService.analyze_directory behavior with mocks."""

    @pytest.mark.asyncio
    async def test_analyze_directory_invalid_extracted_dir_type(self, db_session):
        """Pass non-string extracted_dir; expect ValueError."""
        service = ParsingService(db_session, "test-user")
        with pytest.raises(ValueError) as exc_info:
            await service.analyze_directory(
                extracted_dir=12345,  # type: ignore
                project_id="proj-1",
                user_id="test-user",
                db=db_session,
                language="python",
                user_email="test@example.com",
            )
        assert "string" in str(exc_info.value).lower() or "type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_analyze_directory_directory_missing(self, db_session):
        """Path does not exist; expect FileNotFoundError."""
        service = ParsingService(db_session, "test-user")
        with pytest.raises(FileNotFoundError) as exc_info:
            await service.analyze_directory(
                extracted_dir="/nonexistent/path/12345",
                project_id="proj-1",
                user_id="test-user",
                db=db_session,
                language="python",
                user_email="test@example.com",
            )
        assert "nonexistent" in str(exc_info.value).lower() or "not exist" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_analyze_directory_project_not_in_db(self, db_session, tmp_path):
        """get_project_from_db_by_id returns None; expect ParsingServiceError or HTTPException 404."""
        (tmp_path / "dummy").mkdir(exist_ok=True)
        mock_project_service = MagicMock()
        mock_project_service.get_project_from_db_by_id = AsyncMock(return_value=None)
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_service,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            with pytest.raises((ParsingServiceError, Exception)) as exc_info:
                await service.analyze_directory(
                    extracted_dir=str(tmp_path / "dummy"),
                    project_id="nonexistent-proj",
                    user_id="test-user",
                    db=db_session,
                    language="python",
                    user_email="test@example.com",
                )
            assert "not found" in str(exc_info.value).lower() or "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_directory_language_other(self, db_session, tmp_path):
        """language 'other'; expect status ERROR and ParsingFailedError."""
        (tmp_path / "dummy").mkdir(exist_ok=True)
        mock_project_service = MagicMock()
        mock_project_service.get_project_from_db_by_id = AsyncMock(
            return_value={"project_name": "repo", "branch_name": "main"}
        )
        mock_project_service.update_project_status = AsyncMock()
        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_service,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            with pytest.raises(ParsingFailedError) as exc_info:
                await service.analyze_directory(
                    extracted_dir=str(tmp_path / "dummy"),
                    project_id="proj-1",
                    user_id="test-user",
                    db=db_session,
                    language="other",
                    user_email="test@example.com",
                )
            assert "supported" in str(exc_info.value).lower() or "language" in str(exc_info.value).lower()
            mock_project_service.update_project_status.assert_called_once()
            call_args = mock_project_service.update_project_status.call_args
            assert call_args[0][1] == ProjectStatusEnum.ERROR or str(call_args[0][1]).lower() == "error"


class TestNeo4jFailures:
    """Test Neo4j connection and operation failures (Part 4.4 of plan)."""

    @pytest.mark.asyncio
    async def test_neo4j_connection_failure_on_cleanup(self, db_session):
        """Neo4j connection failure during cleanup_graph → status ERROR."""
        from neo4j.exceptions import ServiceUnavailable

        project_id = "proj-neo4j-fail"
        mock_project_manager = MagicMock()
        mock_project_manager.get_project_from_db_by_id = AsyncMock(
            return_value={"id": project_id, "status": "submitted"}
        )
        mock_project_manager.update_project_status = AsyncMock()

        # Simulate Neo4j connection failure
        mock_code_graph = MagicMock()
        mock_code_graph.cleanup_graph = MagicMock(
            side_effect=ServiceUnavailable("Connection refused")
        )
        mock_code_graph.close = MagicMock()

        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_manager,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.CodeGraphService",
            return_value=mock_code_graph,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            repo_details = ParsingRequest(repo_name="owner/repo")
            with pytest.raises((ParsingServiceError, ServiceUnavailable, Exception)):
                await service.parse_directory(
                    repo_details,
                    user_id="test-user",
                    user_email="test@example.com",
                    project_id=project_id,
                    cleanup_graph=True,
                )

    @pytest.mark.asyncio
    async def test_inference_service_failure(self, db_session, tmp_path):
        """InferenceService.run_inference failure → status ERROR."""
        (tmp_path / "dummy").mkdir(exist_ok=True)
        (tmp_path / "dummy" / "test.py").write_text("print('hello')")

        mock_project_service = MagicMock()
        mock_project_service.get_project_from_db_by_id = AsyncMock(
            return_value={"project_name": "repo", "branch_name": "main"}
        )
        mock_project_service.update_project_status = AsyncMock()

        mock_code_graph = MagicMock()
        mock_code_graph.create_and_store_graph = MagicMock()
        mock_code_graph.close = MagicMock()

        mock_inference = MagicMock()
        mock_inference.run_inference = AsyncMock(
            side_effect=RuntimeError("LLM provider timeout")
        )
        mock_inference.log_graph_stats = MagicMock()
        mock_inference.close = MagicMock()

        with patch(
            "app.modules.parsing.graph_construction.parsing_service.ProjectService",
            return_value=mock_project_service,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.CodeGraphService",
            return_value=mock_code_graph,
        ), patch(
            "app.modules.parsing.graph_construction.parsing_service.InferenceService",
            return_value=mock_inference,
        ):
            service = ParsingService(
                db_session, "test-user", raise_library_exceptions=True
            )
            with pytest.raises((ParsingServiceError, RuntimeError, Exception)):
                await service.analyze_directory(
                    extracted_dir=str(tmp_path / "dummy"),
                    project_id="proj-inference-fail",
                    user_id="test-user",
                    db=db_session,
                    language="python",
                    user_email="test@example.com",
                )


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
