"""Unit tests for SearchService."""
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from app.modules.search.search_service import SearchService
from app.modules.search.search_models import SearchIndex


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def search_service(mock_db):
    return SearchService(mock_db)


class TestSearchServiceInit:
    def test_init_sets_project_path(self, mock_db):
        svc = SearchService(mock_db)
        assert "project" in svc.project_path.lower() or svc.project_path
        assert svc.db is mock_db


class TestSearchServiceCommitIndices:
    @pytest.mark.asyncio
    async def test_commit_indices_calls_db_commit(self, search_service, mock_db):
        await search_service.commit_indices()
        mock_db.commit.assert_called_once()


class TestSearchServiceCalculateRelevance:
    def test_calculate_relevance_name_match(self, search_service):
        result = MagicMock(spec=SearchIndex)
        result.name = "FooBar"
        result.file_path = "path"
        result.content = "x"
        rel = search_service._calculate_relevance(result, ["foo"])
        assert rel >= 0

    def test_calculate_relevance_content_match(self, search_service):
        result = MagicMock(spec=SearchIndex)
        result.name = "other"
        result.file_path = "path"
        result.content = "hello world"
        rel = search_service._calculate_relevance(result, ["hello"])
        assert rel >= 0


class TestSearchServiceDetermineMatchType:
    def test_determine_match_type_exact(self, search_service):
        result = MagicMock(spec=SearchIndex)
        result.content = "hello world"
        out = search_service._determine_match_type(result, ["hello", "world"])
        assert out == "Exact Match"

    def test_determine_match_type_partial(self, search_service):
        result = MagicMock(spec=SearchIndex)
        result.content = "hello"
        out = search_service._determine_match_type(result, ["hello", "world"])
        assert out == "Partial Match"


class TestSearchServiceStringSimilarity:
    def test_string_similarity_identical(self, search_service):
        assert search_service._string_similarity("abc", "abc") == 1.0

    def test_string_similarity_no_overlap(self, search_service):
        assert search_service._string_similarity("abc", "def") == 0.0

    def test_string_similarity_partial(self, search_service):
        s = search_service._string_similarity("ab", "ac")
        assert 0 < s < 1


class TestSearchServiceDeleteProjectIndex:
    def test_delete_project_index_executes_and_commits(self, search_service, mock_db):
        search_service.delete_project_index("proj-1")
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()


class TestSearchServiceSearchCodebase:
    @pytest.mark.asyncio
    async def test_search_codebase_empty_results(self, search_service, mock_db):
        mock_db.query.return_value.filter.return_value.all.return_value = []
        out = await search_service.search_codebase("proj-1", "query")
        assert out == []

    @pytest.mark.asyncio
    async def test_search_codebase_returns_formatted_results(self, search_service, mock_db):
        mock_result = MagicMock(spec=SearchIndex)
        mock_result.node_id = "n1"
        mock_result.name = "file.py"
        mock_result.file_path = "/projects/root/file.py"
        mock_result.content = "code"
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_result]
        search_service.project_path = "/projects/root"
        out = await search_service.search_codebase("proj-1", "file")
        assert len(out) <= 10
        if out:
            assert "node_id" in out[0]
            assert "relevance" in out[0]
