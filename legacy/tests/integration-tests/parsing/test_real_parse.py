"""
Real parse test: runs ParsingService.parse_directory with real Postgres, Neo4j, and a local repo path.

This test catches regressions in core parsing logic that mocked tests miss.
Requires:
  - Postgres running (POSTGRES_SERVER env var)
  - Neo4j running (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD env vars)

Run with: uv run pytest tests/integration-tests/parsing/test_real_parse.py -v -m real_parse
Skip with: uv run pytest -m "not real_parse"
"""

import uuid
import pytest

from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.projects.projects_service import ProjectService
from app.modules.projects.projects_schema import ProjectStatusEnum


pytestmark = [pytest.mark.real_parse, pytest.mark.asyncio]


class TestRealParse:
    """Real parsing tests with actual Postgres + Neo4j + local repo."""

    async def test_parse_local_repo_succeeds(
        self,
        db_session,
        neo4j_config,
        test_repo_path,
        setup_test_user_committed,
    ):
        """Parse a local test repo and verify project reaches READY status."""
        user_id = setup_test_user_committed.uid
        user_email = setup_test_user_committed.email or "test@example.com"
        project_id = str(uuid.uuid4())

        project_service = ProjectService(db_session)
        await project_service.register_project(
            repo_name=test_repo_path.split("/")[-1],
            branch_name="",
            user_id=user_id,
            project_id=project_id,
            repo_path=test_repo_path,
        )
        db_session.commit()

        repo_details = ParsingRequest(repo_path=test_repo_path)

        parsing_service = ParsingService.create_from_config(
            db=db_session,
            user_id=user_id,
            neo4j_config=neo4j_config,
            raise_library_exceptions=True,
        )

        result = await parsing_service.parse_directory(
            repo_details=repo_details,
            user_id=user_id,
            user_email=user_email,
            project_id=project_id,
            cleanup_graph=True,
        )

        assert result is not None, "parse_directory should return a result"

        project = await project_service.get_project_from_db_by_id(project_id)
        assert project is not None, "Project should exist in DB after parsing"

        status = project.get("status")
        assert status in (
            ProjectStatusEnum.READY.value,
            ProjectStatusEnum.PARSED.value,
            "ready",
            "parsed",
        ), f"Expected READY or PARSED status, got {status}"

    async def test_parse_local_repo_detects_python(
        self,
        db_session,
        neo4j_config,
        test_repo_path,
        setup_test_user_committed,
    ):
        """Parse verifies the test repo is detected as Python."""
        from app.modules.parsing.graph_construction.parsing_helper import ParseHelper

        detected = ParseHelper.detect_repo_language(test_repo_path)
        assert detected == "python", f"Expected 'python', got '{detected}'"

    async def test_parse_creates_graph_nodes(
        self,
        db_session,
        neo4j_config,
        test_repo_path,
        setup_test_user_committed,
    ):
        """Parse creates at least some nodes in Neo4j for the project."""
        from neo4j import GraphDatabase

        user_id = setup_test_user_committed.uid
        user_email = setup_test_user_committed.email or "test@example.com"
        project_id = str(uuid.uuid4())

        project_service = ProjectService(db_session)
        await project_service.register_project(
            repo_name=test_repo_path.split("/")[-1],
            branch_name="",
            user_id=user_id,
            project_id=project_id,
            repo_path=test_repo_path,
        )
        db_session.commit()

        repo_details = ParsingRequest(repo_path=test_repo_path)

        parsing_service = ParsingService.create_from_config(
            db=db_session,
            user_id=user_id,
            neo4j_config=neo4j_config,
            raise_library_exceptions=True,
        )

        await parsing_service.parse_directory(
            repo_details=repo_details,
            user_id=user_id,
            user_email=user_email,
            project_id=project_id,
            cleanup_graph=True,
        )

        driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH (n) WHERE n.repoId = $project_id RETURN count(n) AS cnt",
                    project_id=project_id,
                )
                record = result.single()
                node_count = record["cnt"] if record else 0

            assert node_count > 0, (
                f"Expected at least some nodes for project {project_id}, got {node_count}"
            )
        finally:
            driver.close()
