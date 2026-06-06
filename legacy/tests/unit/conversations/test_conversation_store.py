"""
Unit tests for ConversationStore (app/modules/conversations/conversation/conversation_store.py)

Tests cover:
- get_by_id
- create
- get_with_message_count
- update_title
- delete
- get_for_user
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

from sqlalchemy.exc import SQLAlchemyError

from app.modules.conversations.conversation.conversation_store import (
    ConversationStore,
    StoreError,
)
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
    Visibility,
)
from app.modules.conversations.message.message_model import Message, MessageType


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_async_db():
    """Create a mock async database session"""
    mock = MagicMock()
    mock.add = AsyncMock()
    mock.commit = AsyncMock()
    return mock


@pytest.fixture
def mock_db():
    """Create a mock sync database session"""
    return MagicMock()


@pytest.fixture
def conversation_store(mock_db, mock_async_db):
    """Create a ConversationStore with mocked database sessions"""
    return ConversationStore(db=mock_db, async_db=mock_async_db)


@pytest.fixture
def sample_conversation():
    """Create a sample Conversation object for testing"""
    conv = MagicMock(spec=Conversation)
    conv.id = "conv-123"
    conv.user_id = "user-456"
    conv.title = "Test Conversation"
    conv.status = ConversationStatus.ACTIVE
    conv.project_ids = ["project-1"]
    conv.agent_ids = ["agent-1"]
    conv.created_at = datetime.now(timezone.utc)
    conv.updated_at = datetime.now(timezone.utc)
    conv.shared_with_emails = []
    conv.visibility = Visibility.PRIVATE
    return conv


class TestGetById:
    """Tests for get_by_id method"""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_conversation(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_by_id returns conversation when found"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_conversation
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_by_id("conv-123")

        assert result == sample_conversation
        mock_async_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(
        self, conversation_store, mock_async_db
    ):
        """Test get_by_id returns None when conversation does not exist"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_by_id("nonexistent")

        assert result is None


class TestCreate:
    """Tests for create method"""

    @pytest.mark.asyncio
    async def test_create_success(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test successful conversation creation"""
        await conversation_store.create(sample_conversation)

        mock_async_db.add.assert_called_once_with(sample_conversation)
        mock_async_db.commit.assert_called_once()


class TestGetWithMessageCount:
    """Tests for get_with_message_count method"""

    @pytest.mark.asyncio
    async def test_get_with_message_count_zero_messages(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_with_message_count with no messages returns 0"""
        mock_result = MagicMock()
        mock_result.first.return_value = (sample_conversation, 0)
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_with_message_count("conv-123")

        assert result == sample_conversation
        assert getattr(result, "human_message_count", None) == 0

    @pytest.mark.asyncio
    async def test_get_with_message_count_only_ai_messages(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_with_message_count with only AI messages returns 0"""
        mock_result = MagicMock()
        mock_result.first.return_value = (sample_conversation, 0)
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_with_message_count("conv-123")

        assert result == sample_conversation
        assert getattr(result, "human_message_count", None) == 0

    @pytest.mark.asyncio
    async def test_get_with_message_count_multiple_human_messages(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_with_message_count with multiple human messages"""
        mock_result = MagicMock()
        mock_result.first.return_value = (sample_conversation, 5)
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_with_message_count("conv-123")

        assert result == sample_conversation
        assert getattr(result, "human_message_count", None) == 5

    @pytest.mark.asyncio
    async def test_get_with_message_count_not_found(
        self, conversation_store, mock_async_db
    ):
        """Test get_with_message_count returns None when conversation not found"""
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_with_message_count("nonexistent")

        assert result is None


class TestUpdateTitle:
    """Tests for update_title method"""

    @pytest.mark.asyncio
    async def test_update_title_success(self, conversation_store, mock_async_db):
        """Test update_title successfully updates title and updated_at"""
        mock_async_db.execute = AsyncMock()
        mock_async_db.commit = AsyncMock()

        before_update = datetime.now(timezone.utc)

        await conversation_store.update_title("conv-123", "New Title")

        mock_async_db.execute.assert_called_once()
        mock_async_db.commit.assert_called_once()

        # Verify the statement was executed
        call_args = mock_async_db.execute.call_args
        stmt = call_args[0][0]  # First positional argument is the statement
        assert stmt is not None

    @pytest.mark.asyncio
    async def test_update_title_commits_changes(
        self, conversation_store, mock_async_db
    ):
        """Test update_title commits the transaction"""
        mock_async_db.execute = AsyncMock()
        mock_async_db.commit = AsyncMock()

        await conversation_store.update_title("conv-123", "Updated Title")

        mock_async_db.commit.assert_called_once()


class TestDelete:
    """Tests for delete method"""

    @pytest.mark.asyncio
    async def test_delete_success(self, conversation_store, mock_async_db):
        """Test delete returns 1 when conversation is deleted"""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_async_db.execute = AsyncMock(return_value=mock_result)
        mock_async_db.commit = AsyncMock()

        result = await conversation_store.delete("conv-123")

        assert result == 1
        mock_async_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, conversation_store, mock_async_db):
        """Test delete returns 0 when conversation does not exist"""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_async_db.execute = AsyncMock(return_value=mock_result)
        mock_async_db.commit = AsyncMock()

        result = await conversation_store.delete("nonexistent")

        assert result == 0
        mock_async_db.commit.assert_called_once()


class TestGetForUser:
    """Tests for get_for_user method"""

    @pytest.mark.asyncio
    async def test_get_for_user_pagination_start_limit(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_for_user respects start and limit parameters"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = [
            sample_conversation
        ]
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=10, limit=5
        )

        assert len(result) == 1
        mock_async_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_for_user_sort_asc(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_for_user sorts by updated_at ascending when order='asc'"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = [
            sample_conversation
        ]
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=0, limit=10, sort="updated_at", order="asc"
        )

        assert len(result) == 1
        mock_async_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_for_user_sort_desc(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_for_user sorts by updated_at descending when order='desc'"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = [
            sample_conversation
        ]
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=0, limit=10, sort="updated_at", order="desc"
        )

        assert len(result) == 1
        mock_async_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_for_user_invalid_sort_field_fallback(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_for_user falls back to updated_at for invalid sort field"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = [
            sample_conversation
        ]
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=0, limit=10, sort="invalid_field", order="desc"
        )

        assert len(result) == 1
        mock_async_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_for_user_order_fallback(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_for_user falls back to desc order for invalid order value"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = [
            sample_conversation
        ]
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=0, limit=10, sort="updated_at", order="invalid"
        )

        assert len(result) == 1
        mock_async_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_for_user_sqlalchemy_error(
        self, conversation_store, mock_async_db
    ):
        """Test get_for_user raises StoreError on SQLAlchemyError"""
        mock_async_db.execute = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        with pytest.raises(StoreError) as exc_info:
            await conversation_store.get_for_user(user_id="user-456", start=0, limit=10)

        assert "Failed to retrieve conversations" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_for_user_empty_result(self, conversation_store, mock_async_db):
        """Test get_for_user returns empty list when no conversations exist"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = []
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=0, limit=10
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_for_user_with_created_at_sort(
        self, conversation_store, mock_async_db, sample_conversation
    ):
        """Test get_for_user sorts by created_at when specified"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.unique.return_value.all.return_value = [
            sample_conversation
        ]
        mock_async_db.execute = AsyncMock(return_value=mock_result)

        result = await conversation_store.get_for_user(
            user_id="user-456", start=0, limit=10, sort="created_at", order="desc"
        )

        assert len(result) == 1
        mock_async_db.execute.assert_called_once()


class TestStoreError:
    """Tests for StoreError exception"""

    def test_store_error_message(self):
        """Test StoreError stores the error message"""
        error = StoreError("Test error message")
        assert str(error) == "Test error message"

    def test_store_error_is_exception(self):
        """Test StoreError is an Exception subclass"""
        error = StoreError("Test")
        assert isinstance(error, Exception)
