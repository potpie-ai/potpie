"""Unit tests for UserController."""
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.modules.users.user_controller import UserController


pytestmark = pytest.mark.unit


class TestUserControllerGetUserProfilePic:
    @pytest.mark.asyncio
    async def test_get_user_profile_pic_returns_service_result(self):
        mock_db = MagicMock()
        mock_service = MagicMock()
        mock_service.get_user_profile_pic = AsyncMock(
            return_value={"user_id": "u1", "profile_pic_url": "https://example.com/photo.jpg"}
        )
        with patch("app.modules.users.user_controller.UserService", return_value=mock_service):
            ctrl = UserController(mock_db)
            result = await ctrl.get_user_profile_pic("u1")
            assert result["user_id"] == "u1"
            assert result["profile_pic_url"] == "https://example.com/photo.jpg"
            mock_service.get_user_profile_pic.assert_called_once_with("u1")
