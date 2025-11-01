import pytest
from pydantic import ValidationError

from app.modules.search.search_schema import SearchRequest


def test_search_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchRequest(project_id="proj", query="")


def test_search_request_rejects_whitespace_query():
    with pytest.raises(ValidationError):
        SearchRequest(project_id="proj", query="   \t\n ")
