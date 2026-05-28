"""
Unit tests for ConversationService (app/modules/conversations/conversation/conversation_service.py)

Tests cover:
- check_conversation_access
- create_conversation
- get_conversations
- get_conversation_info
- delete_conversation
- conversation message handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

from app.modules.conversations.conversation.conversation_service import (
    ConversationService,
    ConversationServiceError,
    ConversationNotFoundError,
)
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
    Visibility,
)
from app.modules.conversations.conversation.conversation_schema import (
    ConversationAccessType,
    CreateConversationRequest,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_services():
    """Create mock services for ConversationService"""
    return {
        "db": MagicMock(),
        "conversation_store": MagicMock(),
        "message_store": MagicMock(),
        "project_service": MagicMock(),
        "history_manager": MagicMock(),
        "provider_service": MagicMock(),
        "tools_service": MagicMock(),
        "prompt_service": MagicMock(),
        "agent_service": MagicMock(),
        "custom_agent_service": MagicMock(),
        "media_service": MagicMock(),
        "session_service": MagicMock(),
        "redis_manager": MagicMock(),
    }


@pytest.fixture
def conversation_service(mock_services):
    """Create a ConversationService with mocked dependencies"""
    service = ConversationService(
        db=mock_services["db"],
        user_id="test-user",
        user_email="test@example.com",
        conversation_store=mock_services["conversation_store"],
        message_store=mock_services["message_store"],
        project_service=mock_services["project_service"],
        history_manager=mock_services["history_manager"],
        provider_service=mock_services["provider_service"],
        tools_service=mock_services["tools_service"],
        promt_service=mock_services["prompt_service"],
        agent_service=mock_services["agent_service"],
        custom_agent_service=mock_services["custom_agent_service"],
        media_service=mock_services["media_service"],
        session_service=mock_services["session_service"],
        redis_manager=mock_services["redis_manager"],
    )
    return service


class TestCheckConversationAccess:
    """Tests for check_conversation_access"""

    @pytest.mark.asyncio
    async def test_access_no_email_returns_write(self, conversation_service):
        """Test that empty email returns WRITE access"""
        result = await conversation_service.check_conversation_access(
            "conv-123", "", None
        )
        assert result == ConversationAccessType.WRITE

    @pytest.mark.asyncio
    async def test_access_creator_has_write(self, conversation_service, mock_services):
        """Test that conversation creator has WRITE access"""
        mock_conversation = MagicMock()
        mock_conversation.user_id = "test-user"
        mock_conversation.visibility = Visibility.PRIVATE

        mock_services["conversation_store"].get_by_id = AsyncMock(
            return_value=mock_conversation
        )

        result = await conversation_service.check_conversation_access(
            "conv-123", "test@example.com", "test-user"
        )
        assert result == ConversationAccessType.WRITE

    @pytest.mark.asyncio
    async def test_access_public_conversation(self, conversation_service, mock_services):
        """Test that public conversation gives READ access to non-creators"""
        mock_conversation = MagicMock()
        mock_conversation.user_id = "other-user"
        mock_conversation.visibility = Visibility.PUBLIC

        mock_services["conversation_store"].get_by_id = AsyncMock(
            return_value=mock_conversation
        )

        result = await conversation_service.check_conversation_access(
            "conv-123", "viewer@example.com", "viewer-user"
        )
        assert result == ConversationAccessType.READ

    @pytest.mark.asyncio
    async def test_access_not_found(self, conversation_service, mock_services):
        """Test that non-existent conversation returns NOT_FOUND"""
        mock_services["conversation_store"].get_by_id = AsyncMock(return_value=None)

        result = await conversation_service.check_conversation_access(
            "nonexistent", "test@example.com", "test-user"
        )
        assert result == ConversationAccessType.NOT_FOUND


class TestCreateConversation:
    """Tests for create_conversation"""

    @pytest.mark.asyncio
    async def test_create_conversation_invalid_agent(
        self, conversation_service, mock_services
    ):
        """Test create_conversation raises error for invalid agent"""
        mock_services["agent_service"].validate_agent_id = AsyncMock(return_value=False)

        request = CreateConversationRequest(
            user_id="test-user",
            title="Test Conversation",
            status="active",
            project_ids=["project-1"],
            agent_ids=["invalid-agent"],
        )

        # The outer exception wraps the inner "Invalid agent_id" error
        with pytest.raises(ConversationServiceError):
            await conversation_service.create_conversation(request, "test-user")

    @pytest.mark.asyncio
    async def test_create_conversation_success(
        self, conversation_service, mock_services
    ):
        """Test successful conversation creation"""
        mock_services["agent_service"].validate_agent_id = AsyncMock(return_value=True)
        mock_services["project_service"].get_project_name = AsyncMock(
            return_value="Test Project"
        )

        # Mock _create_conversation_record
        with patch.object(
            conversation_service,
            "_create_conversation_record",
            new_callable=AsyncMock,
            return_value="conv-new-123",
        ):
            # Mock _add_system_message
            with patch.object(
                conversation_service,
                "_add_system_message",
                new_callable=AsyncMock,
            ):
                # Also mock CodeProviderService to avoid actual DB call
                with patch(
                    "app.modules.conversations.conversation.conversation_service.CodeProviderService"
                ):
                    request = CreateConversationRequest(
                        user_id="test-user",
                        title="Untitled",
                        status="active",
                        project_ids=["project-1"],
                        agent_ids=["default-chat-agent"],
                    )

                    conv_id, message = await conversation_service.create_conversation(
                        request, "test-user"
                    )

                    assert conv_id == "conv-new-123"
                    assert "successfully" in message.lower()


class TestGetConversationsWithProjects:
    """Tests for get_conversations_with_projects_for_user"""

    @pytest.mark.asyncio
    async def test_get_conversations_empty(self, conversation_service, mock_services):
        """Test getting conversations when none exist"""
        mock_services["conversation_store"].get_for_user = AsyncMock(return_value=[])

        result = await conversation_service.get_conversations_with_projects_for_user(
            user_id="test-user", start=0, limit=10
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_conversations_with_results(
        self, conversation_service, mock_services
    ):
        """Test getting conversations returns correct data"""
        mock_conv = MagicMock()
        mock_conv.id = "conv-1"
        mock_conv.title = "Test Conversation"
        mock_conv.status = ConversationStatus.ACTIVE
        mock_conv.created_at = datetime.now(timezone.utc)
        mock_conv.updated_at = datetime.now(timezone.utc)
        mock_conv.project_ids = ["project-1"]
        mock_conv.agent_ids = ["agent-1"]

        mock_services["conversation_store"].get_for_user = AsyncMock(
            return_value=[mock_conv]
        )

        result = await conversation_service.get_conversations_with_projects_for_user(
            user_id="test-user", start=0, limit=10
        )

        assert len(result) == 1
        assert result[0].id == "conv-1"


class TestDeleteConversation:
    """Tests for delete_conversation"""

    @pytest.mark.asyncio
    async def test_delete_conversation_success(
        self, conversation_service, mock_services
    ):
        """Test successful conversation deletion"""
        # Mock access check to return WRITE
        with patch.object(
            conversation_service,
            "check_conversation_access",
            new_callable=AsyncMock,
            return_value=ConversationAccessType.WRITE,
        ):
            mock_services["message_store"].delete_for_conversation = AsyncMock(
                return_value=5
            )
            mock_services["conversation_store"].delete = AsyncMock(return_value=1)

            result = await conversation_service.delete_conversation(
                "conv-123", "test-user"
            )

            assert result is not None
            mock_services["conversation_store"].delete.assert_called_once_with("conv-123")

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(
        self, conversation_service, mock_services
    ):
        """Test deleting non-existent conversation"""
        with patch.object(
            conversation_service,
            "check_conversation_access",
            new_callable=AsyncMock,
            return_value=ConversationAccessType.WRITE,
        ):
            mock_services["message_store"].delete_for_conversation = AsyncMock(
                return_value=0
            )
            mock_services["conversation_store"].delete = AsyncMock(return_value=0)

            with pytest.raises(ConversationNotFoundError):
                await conversation_service.delete_conversation(
                    "nonexistent", "test-user"
                )


class TestGetConversationInfo:
    """Tests for get_conversation_info"""

    @pytest.mark.asyncio
    async def test_get_conversation_info_not_found(
        self, conversation_service, mock_services
    ):
        """Test getting info for non-existent conversation raises error"""
        mock_services["conversation_store"].get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ConversationNotFoundError):
            await conversation_service.get_conversation_info("nonexistent", "test-user")

    @pytest.mark.asyncio
    async def test_get_conversation_info_success(
        self, conversation_service, mock_services
    ):
        """Test getting conversation info successfully"""
        mock_conv = MagicMock()
        mock_conv.id = "conv-123"
        mock_conv.title = "My Conversation"
        mock_conv.status = ConversationStatus.ACTIVE
        mock_conv.user_id = "test-user"
        mock_conv.created_at = datetime.now(timezone.utc)
        mock_conv.updated_at = datetime.now(timezone.utc)
        mock_conv.project_ids = ["project-1"]
        mock_conv.agent_ids = ["system-agent-1"]
        mock_conv.visibility = Visibility.PRIVATE
        mock_conv.shared_with_emails = []
        mock_conv.total_tokens = 1000
        mock_conv.model_type = "gpt-4"

        mock_services["conversation_store"].get_by_id = AsyncMock(return_value=mock_conv)
        mock_services["message_store"].count_active_for_conversation = AsyncMock(
            return_value=10
        )
        mock_services["project_service"].get_project_name_from_db = AsyncMock(
            return_value="Test Project"
        )

        # Mock check_conversation_access
        with patch.object(
            conversation_service,
            "check_conversation_access",
            new_callable=AsyncMock,
            return_value=ConversationAccessType.WRITE,
        ):
            # Mock agent_service._system_agents to return agent
            mock_services["agent_service"]._system_agents = MagicMock(
                return_value={"system-agent-1": MagicMock()}
            )

            result = await conversation_service.get_conversation_info(
                "conv-123", "test-user"
            )

            assert result.id == "conv-123"
            assert result.title == "My Conversation"


class TestHistoryMethods:
    """Tests for history manager dispatch methods"""

    @pytest.mark.asyncio
    async def test_history_get_session_history_sync(
        self, conversation_service, mock_services
    ):
        """Test dispatching to sync history manager"""
        mock_services["history_manager"].get_session_history.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        result = await conversation_service._history_get_session_history(
            "test-user", "conv-123"
        )

        assert len(result) == 1
        mock_services["history_manager"].get_session_history.assert_called_once()

    def test_history_add_message_chunk(self, conversation_service, mock_services):
        """Test adding message chunk to history"""
        from app.modules.conversations.message.message_model import MessageType

        conversation_service._history_add_message_chunk(
            conversation_id="conv-123",
            content="Hello",
            message_type=MessageType.HUMAN,
            sender_id="test-user",
        )

        mock_services["history_manager"].add_message_chunk.assert_called_once()


class TestParseStrToMessage:
    """Tests for parse_str_to_message (JSON chunk to ChatMessageResponse)"""

    def test_parse_valid_chunk(self, conversation_service):
        chunk = '{"message": "Hello", "citations": ["c1"], "tool_calls": []}'
        result = conversation_service.parse_str_to_message(chunk)
        assert result.message == "Hello"
        assert result.citations == ["c1"]
        assert result.tool_calls == []

    def test_parse_chunk_with_tool_calls(self, conversation_service):
        chunk = '{"message": "Done", "citations": [], "tool_calls": [{"name": "search"}]}'
        result = conversation_service.parse_str_to_message(chunk)
        assert result.message == "Done"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_parse_invalid_json_raises(self, conversation_service):
        with pytest.raises(ConversationServiceError, match="Failed to parse AI response"):
            conversation_service.parse_str_to_message("not json")

    def test_parse_empty_message_defaults(self, conversation_service):
        chunk = "{}"
        result = conversation_service.parse_str_to_message(chunk)
        assert result.message == ""
        assert result.citations == []
        assert result.tool_calls == []


class TestConversationServiceError:
    """Tests for exception classes"""

    def test_conversation_service_error(self):
        """Test ConversationServiceError"""
        error = ConversationServiceError("Test error")
        assert str(error) == "Test error"

    def test_conversation_not_found_error(self):
        """Test ConversationNotFoundError"""
        error = ConversationNotFoundError("Conversation not found")
        assert isinstance(error, ConversationServiceError)
