"""
Unit tests for ShareChatService and AsyncShareChatService (access module).
"""

import pytest
from datetime import datetime, timezone
from fastapi import HTTPException

from app.modules.conversations.access.access_service import (
    ShareChatService,
    AsyncShareChatService,
    ShareChatServiceError,
)
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    Visibility,
    ConversationStatus,
)
from app.modules.users.user_model import User


pytestmark = pytest.mark.unit


@pytest.fixture
def share_chat_user(db_session):
    """User who owns the conversation for share tests."""
    import uuid
    uid = f"share-user-{uuid.uuid4().hex[:8]}"
    user = User(
        uid=uid,
        email=f"share-{uid}@example.com",
        display_name="Share User",
        email_verified=True,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def share_chat_conversation(db_session, share_chat_user):
    """Conversation for share-chat tests."""
    import uuid
    convo = Conversation(
        id=f"share-convo-{uuid.uuid4().hex[:8]}",
        user_id=share_chat_user.uid,
        title="Share Test",
        status=ConversationStatus.ACTIVE,
        project_ids=[],
        agent_ids=["default"],
        visibility=Visibility.PRIVATE,
        shared_with_emails=None,
    )
    db_session.add(convo)
    db_session.commit()
    db_session.refresh(convo)
    return convo


class TestShareChatService:
    """ShareChatService (async methods)."""

    @pytest.mark.asyncio
    async def test_share_chat_set_public(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """Setting visibility to PUBLIC commits and returns conversation_id."""
        svc = ShareChatService(db_session)
        result = await svc.share_chat(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
            visibility=Visibility.PUBLIC,
        )
        assert result == share_chat_conversation.id
        db_session.refresh(share_chat_conversation)
        assert share_chat_conversation.visibility == Visibility.PUBLIC

    @pytest.mark.asyncio
    async def test_share_chat_not_found(self, db_session, share_chat_user):
        """Non-existent conversation raises HTTPException 404."""
        svc = ShareChatService(db_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.share_chat(
                conversation_id="nonexistent",
                user_id=share_chat_user.uid,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_share_chat_private_with_emails(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """PRIVATE with recipient_emails adds to shared_with_emails."""
        svc = ShareChatService(db_session)
        result = await svc.share_chat(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
            recipient_emails=["a@example.com", "b@example.com"],
            visibility=Visibility.PRIVATE,
        )
        assert result == share_chat_conversation.id
        db_session.refresh(share_chat_conversation)
        assert set(share_chat_conversation.shared_with_emails or []) == {
            "a@example.com",
            "b@example.com",
        }

    @pytest.mark.asyncio
    async def test_share_chat_invalid_email(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """Invalid email raises ShareChatServiceError (handler wraps HTTPException)."""
        svc = ShareChatService(db_session)
        with pytest.raises(ShareChatServiceError) as exc_info:
            await svc.share_chat(
                conversation_id=share_chat_conversation.id,
                user_id=share_chat_user.uid,
                recipient_emails=["not-an-email"],
                visibility=Visibility.PRIVATE,
            )
        assert "Invalid email" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_shared_emails(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """get_shared_emails returns list (empty or existing)."""
        svc = ShareChatService(db_session)
        emails = await svc.get_shared_emails(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
        )
        assert emails == []

        share_chat_conversation.shared_with_emails = ["x@example.com"]
        db_session.commit()
        emails = await svc.get_shared_emails(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
        )
        assert emails == ["x@example.com"]

    @pytest.mark.asyncio
    async def test_get_shared_emails_not_found(self, db_session, share_chat_user):
        """get_shared_emails 404 for non-existent conversation."""
        svc = ShareChatService(db_session)
        with pytest.raises(HTTPException) as exc_info:
            await svc.get_shared_emails(
                conversation_id="nonexistent",
                user_id=share_chat_user.uid,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_access_success(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """remove_access removes emails and returns True."""
        share_chat_conversation.shared_with_emails = [
            "a@example.com",
            "b@example.com",
        ]
        db_session.commit()

        svc = ShareChatService(db_session)
        result = await svc.remove_access(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
            emails_to_remove=["a@example.com"],
        )
        assert result is True
        db_session.refresh(share_chat_conversation)
        assert share_chat_conversation.shared_with_emails == ["b@example.com"]

    @pytest.mark.asyncio
    async def test_remove_access_no_shared(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """remove_access when no shared access raises ShareChatServiceError."""
        svc = ShareChatService(db_session)
        with pytest.raises(ShareChatServiceError):
            await svc.remove_access(
                conversation_id=share_chat_conversation.id,
                user_id=share_chat_user.uid,
                emails_to_remove=["a@example.com"],
            )

    @pytest.mark.asyncio
    async def test_remove_access_none_have_access(
        self, db_session, share_chat_user, share_chat_conversation
    ):
        """remove_access when specified emails don't have access raises."""
        share_chat_conversation.shared_with_emails = ["a@example.com"]
        db_session.commit()

        svc = ShareChatService(db_session)
        with pytest.raises(ShareChatServiceError):
            await svc.remove_access(
                conversation_id=share_chat_conversation.id,
                user_id=share_chat_user.uid,
                emails_to_remove=["b@example.com"],
            )


class TestAsyncShareChatService:
    """AsyncShareChatService."""

    @pytest.mark.asyncio
    async def test_share_chat_public_async(
        self, db_session, async_db_session, share_chat_user, share_chat_conversation
    ):
        """Async share_chat PUBLIC."""
        svc = AsyncShareChatService(async_db_session)
        result = await svc.share_chat(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
            visibility=Visibility.PUBLIC,
        )
        assert result == share_chat_conversation.id
        emails = await svc.get_shared_emails(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
        )
        assert emails == []  # PUBLIC, no shared emails

    @pytest.mark.asyncio
    async def test_get_shared_emails_async(
        self, db_session, async_db_session, share_chat_user, share_chat_conversation
    ):
        """Async get_shared_emails."""
        share_chat_conversation.shared_with_emails = ["async@example.com"]
        db_session.commit()

        svc = AsyncShareChatService(async_db_session)
        emails = await svc.get_shared_emails(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
        )
        assert emails == ["async@example.com"]

    @pytest.mark.asyncio
    async def test_remove_access_async(
        self, db_session, async_db_session, share_chat_user, share_chat_conversation
    ):
        """Async remove_access."""
        share_chat_conversation.shared_with_emails = ["r1@example.com", "r2@example.com"]
        db_session.commit()

        svc = AsyncShareChatService(async_db_session)
        result = await svc.remove_access(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
            emails_to_remove=["r1@example.com"],
        )
        assert result is True
        emails = await svc.get_shared_emails(
            conversation_id=share_chat_conversation.id,
            user_id=share_chat_user.uid,
        )
        assert set(emails) == {"r2@example.com"}
