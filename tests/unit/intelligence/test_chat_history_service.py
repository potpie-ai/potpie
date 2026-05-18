from types import SimpleNamespace

import pytest

from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
    MessageType,
)
from app.modules.intelligence.memory.chat_history_service import (
    AsyncChatHistoryService,
    ChatHistoryMessage,
    ChatHistoryService,
)


pytestmark = pytest.mark.unit


class _Query:
    """Minimal query double that records chained sync filters."""

    def __init__(self, messages):
        """Initialize the query double with messages to return."""

        self._messages = messages
        self.filters = []
        self.order_by_column = None

    def filter_by(self, **kwargs):
        """Record a filter_by call and keep the query chainable."""

        self.filters.append(kwargs)
        return self

    def order_by(self, column):
        """Record the order_by column and keep the query chainable."""

        self.order_by_column = column
        return self

    def all(self):
        """Return the configured sync query result rows."""

        return self._messages


class _SyncSession:
    """Minimal sync session double for ChatHistoryService queries."""

    def __init__(self, messages):
        """Initialize the sync session with messages to expose."""

        self._messages = messages
        self.query_model = None
        self.last_query = None

    def query(self, model):
        """Record the queried model and return a query double."""

        self.query_model = model
        self.last_query = _Query(self._messages)
        return self.last_query


class _ScalarResult:
    """Minimal scalar result double for async query results."""

    def __init__(self, messages):
        """Initialize the scalar result with messages to return."""

        self._messages = messages

    def all(self):
        """Return the configured async scalar rows."""

        return self._messages


class _ExecuteResult:
    """Minimal execute result double exposing scalars()."""

    def __init__(self, messages):
        """Initialize the execute result with messages to expose."""

        self._messages = messages

    def scalars(self):
        """Return scalar rows for the async service."""

        return _ScalarResult(self._messages)


class _AsyncSession:
    """Minimal async session double for AsyncChatHistoryService."""

    def __init__(self, messages):
        """Initialize the async session with messages to return."""

        self._messages = messages
        self.statement = None

    async def execute(self, statement):
        """Record the SQL statement and return an execute result double."""

        self.statement = statement
        return _ExecuteResult(self._messages)


def _message(message_type, content):
    """Create a message-like object with only the fields under test."""

    return SimpleNamespace(type=message_type, content=content)


def test_sync_session_history_uses_lightweight_messages():
    """Verify sync history returns local messages without LangChain wrappers."""

    session = _SyncSession(
        [
            _message(MessageType.HUMAN, "hello"),
            _message(MessageType.AI_GENERATED, "answer"),
            _message(MessageType.SYSTEM_GENERATED, "system note"),
        ]
    )

    history = ChatHistoryService(session).get_session_history(
        user_id="user-1",
        conversation_id="conversation-1",
    )

    assert history == [
        ChatHistoryMessage(type="human", content="hello"),
        ChatHistoryMessage(type="ai", content="answer"),
        ChatHistoryMessage(type="ai", content="system note"),
    ]
    assert session.query_model is Message
    assert session.last_query.filters == [
        {"conversation_id": "conversation-1"},
        {"status": MessageStatus.ACTIVE},
    ]


@pytest.mark.asyncio
async def test_async_session_history_uses_lightweight_messages():
    """Verify async history returns local messages without LangChain wrappers."""

    session = _AsyncSession(
        [
            _message(MessageType.HUMAN, "question"),
            _message(MessageType.AI_GENERATED, "response"),
        ]
    )

    history = await AsyncChatHistoryService(session).get_session_history(
        user_id="user-1",
        conversation_id="conversation-1",
    )

    assert history == [
        ChatHistoryMessage(type="human", content="question"),
        ChatHistoryMessage(type="ai", content="response"),
    ]
    assert session.statement is not None
