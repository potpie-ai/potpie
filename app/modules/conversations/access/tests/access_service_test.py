import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from app.modules.conversations.access.access_service import (
    ShareChatService,
    ShareChatServiceError,
)
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    Visibility,
)


class TestShareChatService:
    """Test cases for ShareChatService class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def share_chat_service(self, mock_db):
        """Create an instance of ShareChatService for testing."""
        return ShareChatService(mock_db)

    @pytest.fixture
    def mock_conversation(self):
        """Create a mock conversation object."""
        conversation = MagicMock(spec=Conversation)
        conversation.id = "test_conversation_id"
        conversation.user_id = "test_user_id"
        conversation.visibility = Visibility.PRIVATE
        conversation.shared_with_emails = []
        return conversation

    class TestShareChat:
        """Test cases for share_chat functionality."""

        @pytest.mark.asyncio
        async def test_share_chat_valid_emails(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with valid email addresses."""
            mock_db.query().filter_by().first.return_value = mock_conversation

            valid_emails = ["user1@example.com", "user2@example.com"]
            result = await share_chat_service.share_chat(
                conversation_id="test_conversation_id",
                user_id="test_user_id",
                recipient_emails=valid_emails,
                visibility=Visibility.PRIVATE,
            )

            assert result == "test_conversation_id"
            assert mock_conversation.shared_with_emails == valid_emails
            mock_db.commit.assert_called()

        @pytest.mark.asyncio
        async def test_share_chat_invalid_email_format(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with invalid email format."""
            mock_db.query().filter_by().first.return_value = mock_conversation

            invalid_emails = ["invalid-email", "another@invalid"]
            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.share_chat(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    recipient_emails=invalid_emails,
                    visibility=Visibility.PRIVATE,
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_share_chat_empty_email(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with empty email string."""
            mock_db.query().filter_by().first.return_value = mock_conversation

            invalid_emails = [""]
            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.share_chat(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    recipient_emails=invalid_emails,
                    visibility=Visibility.PRIVATE,
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_share_chat_whitespace_email(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with whitespace-only email."""
            mock_db.query().filter_by().first.return_value = mock_conversation

            invalid_emails = ["   "]
            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.share_chat(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    recipient_emails=invalid_emails,
                    visibility=Visibility.PRIVATE,
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_share_chat_mixed_valid_invalid_emails(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with mix of valid and invalid emails."""
            mock_db.query().filter_by().first.return_value = mock_conversation

            mixed_emails = ["valid@example.com", "invalid-email"]
            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.share_chat(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    recipient_emails=mixed_emails,
                    visibility=Visibility.PRIVATE,
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)
            # Ensure no emails were added due to validation failure
            mock_db.commit.assert_not_called()

        @pytest.mark.asyncio
        async def test_share_chat_public_visibility(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with public visibility (emails should be ignored)."""
            mock_db.query().filter_by().first.return_value = mock_conversation

            result = await share_chat_service.share_chat(
                conversation_id="test_conversation_id",
                user_id="test_user_id",
                recipient_emails=None,
                visibility=Visibility.PUBLIC,
            )

            assert result == "test_conversation_id"
            assert mock_conversation.visibility == Visibility.PUBLIC
            mock_db.commit.assert_called()

        @pytest.mark.asyncio
        async def test_share_chat_conversation_not_found(
            self, share_chat_service, mock_db
        ):
            """Test sharing non-existent chat."""
            mock_db.query().filter_by().first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.share_chat(
                    conversation_id="nonexistent_id",
                    user_id="test_user_id",
                    recipient_emails=["test@example.com"],
                    visibility=Visibility.PRIVATE,
                )

            assert exc_info.value.status_code == 404
            assert "does not exist" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_share_chat_duplicate_emails(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test sharing chat with duplicate emails (should only add unique ones)."""
            mock_conversation.shared_with_emails = ["existing@example.com"]
            mock_db.query().filter_by().first.return_value = mock_conversation

            emails = ["existing@example.com", "new@example.com"]
            result = await share_chat_service.share_chat(
                conversation_id="test_conversation_id",
                user_id="test_user_id",
                recipient_emails=emails,
                visibility=Visibility.PRIVATE,
            )

            assert result == "test_conversation_id"
            # Should only add new@example.com, not the duplicate
            assert "new@example.com" in mock_conversation.shared_with_emails
            assert mock_conversation.shared_with_emails.count("existing@example.com") == 1

    class TestRemoveAccess:
        """Test cases for remove_access functionality."""

        @pytest.mark.asyncio
        async def test_remove_access_valid_emails(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access with valid email addresses."""
            mock_conversation.shared_with_emails = [
                "user1@example.com",
                "user2@example.com",
            ]
            mock_db.query().filter_by().first.return_value = mock_conversation

            result = await share_chat_service.remove_access(
                conversation_id="test_conversation_id",
                user_id="test_user_id",
                emails_to_remove=["user1@example.com"],
            )

            assert result is True
            mock_db.commit.assert_called()

        @pytest.mark.asyncio
        async def test_remove_access_invalid_email_format(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access with invalid email format."""
            mock_conversation.shared_with_emails = ["user1@example.com"]
            mock_db.query().filter_by().first.return_value = mock_conversation

            invalid_emails = ["invalid-email"]
            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    emails_to_remove=invalid_emails,
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_remove_access_empty_email(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access with empty email string."""
            mock_conversation.shared_with_emails = ["user1@example.com"]
            mock_db.query().filter_by().first.return_value = mock_conversation

            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    emails_to_remove=[""],
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_remove_access_whitespace_email(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access with whitespace-only email."""
            mock_conversation.shared_with_emails = ["user1@example.com"]
            mock_db.query().filter_by().first.return_value = mock_conversation

            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    emails_to_remove=["   "],
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_remove_access_mixed_valid_invalid_emails(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access with mix of valid and invalid emails."""
            mock_conversation.shared_with_emails = ["user1@example.com"]
            mock_db.query().filter_by().first.return_value = mock_conversation

            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    emails_to_remove=["user1@example.com", "invalid-email"],
                )

            assert exc_info.value.status_code == 400
            assert "Invalid email address" in str(exc_info.value.detail)
            # Ensure no emails were removed due to validation failure
            mock_db.commit.assert_not_called()

        @pytest.mark.asyncio
        async def test_remove_access_conversation_not_found(
            self, share_chat_service, mock_db
        ):
            """Test removing access from non-existent chat."""
            mock_db.query().filter_by().first.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="nonexistent_id",
                    user_id="test_user_id",
                    emails_to_remove=["test@example.com"],
                )

            assert exc_info.value.status_code == 404
            assert "does not exist" in str(exc_info.value.detail)

        @pytest.mark.asyncio
        async def test_remove_access_no_shared_emails(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access when chat has no shared emails."""
            mock_conversation.shared_with_emails = None
            mock_db.query().filter_by().first.return_value = mock_conversation

            with pytest.raises(ShareChatServiceError) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    emails_to_remove=["test@example.com"],
                )

            assert "no shared access to remove" in str(exc_info.value)

        @pytest.mark.asyncio
        async def test_remove_access_email_not_in_shared_list(
            self, share_chat_service, mock_db, mock_conversation
        ):
            """Test removing access for email that doesn't have access."""
            mock_conversation.shared_with_emails = ["user1@example.com"]
            mock_db.query().filter_by().first.return_value = mock_conversation

            with pytest.raises(ShareChatServiceError) as exc_info:
                await share_chat_service.remove_access(
                    conversation_id="test_conversation_id",
                    user_id="test_user_id",
                    emails_to_remove=["nonexistent@example.com"],
                )

            assert "None of the specified emails have access" in str(exc_info.value)