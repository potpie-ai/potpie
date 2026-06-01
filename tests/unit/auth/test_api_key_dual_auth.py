"""Tests for Firebase-or-API-key authentication dependency."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.modules.auth.api_key_deps import get_firebase_or_api_key_user


@pytest.mark.asyncio
async def test_get_firebase_or_api_key_user_prefers_api_key_header() -> None:
    request = MagicMock()
    res = MagicMock()
    expected = {"user_id": "user-1", "email": "a@b.com", "auth_type": "api_key"}

    with patch(
        "app.modules.auth.api_key_deps.get_api_key_user",
        new_callable=AsyncMock,
        return_value=expected,
    ) as mock_api:
        result = await get_firebase_or_api_key_user(
            request,
            res,
            credential=None,
            x_api_key="sk-test",
            x_user_id=None,
            db=MagicMock(),
            async_db=MagicMock(),
        )

    assert result == expected
    mock_api.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_firebase_or_api_key_user_falls_back_to_firebase() -> None:
    request = MagicMock()
    res = MagicMock()
    expected = {"user_id": "firebase-user", "uid": "firebase-user"}

    with patch(
        "app.modules.auth.api_key_deps.AuthService.check_auth",
        new_callable=AsyncMock,
        return_value=expected,
    ) as mock_firebase:
        result = await get_firebase_or_api_key_user(
            request,
            res,
            credential=None,
            x_api_key=None,
            x_user_id=None,
            db=MagicMock(),
            async_db=MagicMock(),
        )

    assert result == expected
    mock_firebase.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_api_key_user_requires_header() -> None:
    from app.modules.auth.api_key_deps import get_api_key_user

    with pytest.raises(HTTPException) as exc:
        await get_api_key_user(
            x_api_key=None,
            x_user_id=None,
            db=MagicMock(),
            async_db=MagicMock(),
        )
    assert exc.value.status_code == 401
