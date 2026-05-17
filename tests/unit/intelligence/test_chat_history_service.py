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
    def __init__(self, messages):
        self._messages = messages
        self.filters = []
        self.order_by_column = None

    def filter_by(self, **kwargs):
        self.filters.append(kwargs)
        return self

    def order_by(self, column):
        self.order_by_column = column
        return self

    def all(self):
        return self._messages


class _SyncSession:
    def __init__(self, messages):
        self._messages = messages
        self.query_model = None
        self.last_query = None

    def query(self, model):
        self.query_model = model
        self.last_query = _Query(self._messages)
        return self.last_query


class _ScalarResult:
    def __init__(self, messages):
        self._messages = messages

    def all(self):
        return self._messages


class _ExecuteResult:
    def __init__(self, messages):
        self._messages = messages

    def scalars(self):
        return _ScalarResult(self._messages)


class _AsyncSession:
    def __init__(self, messages):
        self._messages = messages
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _ExecuteResult(self._messages)


def _message(message_type, content):
    return SimpleNamespace(type=message_type, content=content)


def test_sync_session_history_uses_lightweight_messages():
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
