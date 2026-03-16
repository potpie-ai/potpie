"""
Unit tests for validate_parsing_input decorator.
"""

import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException

from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_validator import validate_parsing_input


pytestmark = pytest.mark.unit


async def _dummy_handler(*args, **kwargs):
    """Dummy async handler for decorator tests."""
    return {"ok": True}


@validate_parsing_input
async def _wrapped_handler(*args, **kwargs):
    return await _dummy_handler(*args, **kwargs)


class TestValidateParsingInput:
    """Test validate_parsing_input decorator."""

    @pytest.mark.asyncio
    async def test_allows_when_dev_mode_and_repo_path(self, monkeypatch):
        """Validator allows request when dev mode enabled and repo_path set."""
        monkeypatch.setenv("isDevelopmentMode", "enabled")
        repo_details = ParsingRequest(repo_path="/tmp/repo")
        result = await _wrapped_handler(
            repo_details=repo_details,
            user_id="any-user",
        )
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_403_when_repo_path_and_non_dev(self, monkeypatch):
        """Validator returns 403 when repo_path set and development mode not enabled."""
        monkeypatch.setenv("isDevelopmentMode", "disabled")
        repo_details = ParsingRequest(repo_path="/tmp/repo")
        with pytest.raises(HTTPException) as exc_info:
            await _wrapped_handler(
                repo_details=repo_details,
                user_id="any-user",
            )
        assert exc_info.value.status_code == 403
        assert "Development mode" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_403_when_default_user_and_repo_name(self, monkeypatch):
        """Validator returns 403 when user_id is defaultUsername and repo_name is set."""
        monkeypatch.setenv("defaultUsername", "default-user")
        monkeypatch.setenv("isDevelopmentMode", "enabled")
        repo_details = ParsingRequest(repo_name="owner/repo")
        with pytest.raises(HTTPException) as exc_info:
            await _wrapped_handler(
                repo_details=repo_details,
                user_id="default-user",
            )
        assert exc_info.value.status_code == 403
        assert "auth token" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_allows_when_user_not_default(self, monkeypatch):
        """Validator allows request when user_id is not defaultUsername."""
        monkeypatch.setenv("defaultUsername", "default-user")
        monkeypatch.setenv("isDevelopmentMode", "enabled")
        repo_details = ParsingRequest(repo_name="owner/repo")
        result = await _wrapped_handler(
            repo_details=repo_details,
            user_id="other-user",
        )
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_passes_through_when_repo_details_missing(self):
        """When repo_details not in kwargs, wrapped handler is invoked."""
        result = await _wrapped_handler(user_id="u1")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_passes_through_when_user_id_missing(self):
        """When user_id not in kwargs, wrapped handler is invoked (no 403)."""
        repo_details = ParsingRequest(repo_name="owner/repo")
        result = await _wrapped_handler(repo_details=repo_details)
        assert result == {"ok": True}
