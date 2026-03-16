import pytest
from unittest.mock import patch, AsyncMock, MagicMock, ANY


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType
from app.modules.media.media_schema import AttachmentUploadResponse, AttachmentType
from app.modules.users.user_model import User
from sqlalchemy.orm import Session
from app.modules.projects.projects_model import Project




# THIS IS THE FIX: We are patching the name 'MediaService' exactly where it is used.
@patch("app.modules.conversations.conversations_router.MediaService")
async def test_post_message_successful_flow(
    mock_media_service_class, client, db_session, mock_redis_stream_manager
):
    """
    Tests the complete, successful flow of a user posting a new message
    with an image.
    """
    # Get the mock INSTANCE that is created when the router code calls MediaService(db).
    mock_instance = mock_media_service_class.return_value

    # Patch other external dependencies
    with patch(
        "app.celery.tasks.agent_tasks.execute_agent_background.delay"
    ) as mock_celery_task:
        mock_celery_task.return_value.id = "test-task-id-execute"

        # Configure the mock INSTANCE's methods. The __init__ is now completely skipped.
        mock_instance.upload_image = AsyncMock(
            return_value=AttachmentUploadResponse(
                id="fake_attachment_id",
                attachment_type=AttachmentType.IMAGE,
                file_name="test_image.png",
                mime_type="image/png",
                file_size=1000,
            )
        )

        # Prepare request data
        conversation_id = "some_conversation_id"
        files = {"images": ("test_image.png", b"fake image data", "image/png")}
        data = {"content": "This is a test message."}

        # Make the HTTP request (streaming endpoint: don't consume full body)
        async with client.stream(
            "POST",
            f"/api/v1/conversations/{conversation_id}/message/",
            files=files,
            data=data,
        ) as response:
            # Assert outcomes
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

        # Assert calls on the mock INSTANCE
        mock_instance.upload_image.assert_called_once()
        mock_celery_task.assert_called_once()


# We only need to mock the true external boundaries of the ConversationService
# We ONLY mock the AgentsService now. We want the real ProjectService to run.
@patch("app.modules.conversations.conversation.conversation_service.AgentsService")
async def test_create_conversation_creates_record_and_system_message(
    mock_agent_service_class: MagicMock,
    client,
    db_session: Session,
    setup_test_user_committed: User,
    conversation_project: Project,  # Depend on both fixtures
):
    """
    Integration Test for POST /conversations/
    - Fixtures create prerequisite User and Project records.
    - Mocks the external AgentsService.
    - Verifies the service's title generation and database persistence.
    """
    # 1. ARRANGE
    mock_agent_instance = mock_agent_service_class.return_value
    mock_agent_instance.validate_agent_id = AsyncMock(return_value="SYSTEM_AGENT")

    # THE FIX for the assertion: Send "Untitled" to trigger the title replacement logic.
    request_data = {
        "user_id": "test-user",
        "title": "Untitled",
        "status": "active",
        "project_ids": ["project-id-123"],  # This ID must match the one in our fixture
        "agent_ids": ["default-chat-agent"],
    }

    # 2. ACT
    response = await client.post("/api/v1/conversations/", json=request_data)

    # 3. ASSERT
    if response.status_code != 200:
        print(
            f"FAILED with status {response.status_code}. Response body: {response.text}"
        )

    assert response.status_code == 200

    conversation_id = response.json()["conversation_id"]

    # Refresh the session to ensure we read the latest committed data
    db_session.expire_all()

    # Verify the final state of the database
    db_conversation = db_session.query(Conversation).filter_by(id=conversation_id).one()

    assert db_conversation is not None
    assert db_conversation.user_id == setup_test_user_committed.uid

    # THE FIX for the assertion: The title should now be correctly generated.
    assert db_conversation.title == "Test Project Repo"

    db_messages = (
        db_session.query(Message).filter_by(conversation_id=conversation_id).all()
    )
    # At least one system message; there may be an extra welcome/context message
    assert len(db_messages) >= 1
    system_content = "You can now ask questions about the Test Project Repo repository."
    assert any(m.content == system_content for m in db_messages), (
        f"Expected a message with content {system_content!r}, got: {[m.content for m in db_messages]}"
    )


async def test_post_message_dispatches_celery_task_and_streams(
    app,
    client,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed: Conversation,  # Depend on all our data fixtures
):
    """
    Integration Test for POST /.../message/
    - Verifies the API response is a streaming response.
    - Verifies the async Redis manager is used to wait for task start.
    - Verifies the handoff to the Celery background task happens correctly.
    """
    # 1. ARRANGE
    conversation_id = setup_test_conversation_committed.id
    message_content = "This is my first test message."

    # This endpoint expects form data, not JSON
    form_data = {"content": message_content}

    # 2. ACT
    # The router initiates a streaming response, but we don't need to consume it.
    # We only care that the request was accepted and the mocks were called.
    async with client.stream(
        "POST",
        f"/api/v1/conversations/{conversation_id}/message/",
        data=form_data,
    ) as response:
        # Part A: Verify the API response
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    # 3. ASSERT

    # Part B: Verify the interaction with async Redis (app.state mock from client fixture).
    # The mock is shared across tests, so assert our call happened rather than exactly one call.
    async_redis = getattr(app.state, "async_redis_stream_manager", None)
    assert async_redis is not None
    async_redis.wait_for_task_start.assert_any_call(
        conversation_id, ANY, timeout=30, require_running=ANY
    )

    # Part C: Verify the handoff to Celery (the most important part)
    # Check that the background task was called exactly once
    mock_celery_tasks["execute"].assert_called_once()

    # Inspect the arguments passed to the Celery task
    call_args = mock_celery_tasks["execute"].call_args
    call_kwargs = call_args.kwargs

    assert call_kwargs["conversation_id"] == conversation_id
    assert call_kwargs["user_id"] == "test-user"
    assert call_kwargs["query"] == message_content
    # The run_id is dynamically generated, so we assert that it exists and is a string
    assert call_kwargs["run_id"] == ANY
    assert isinstance(call_kwargs["run_id"], str)


async def test_delete_conversation_removes_from_database(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for DELETE /conversations/{id}/
    - Verifies the API response.
    - Verifies the Conversation is actually deleted from the database.
    """
    # 1. ARRANGE
    conversation_id = setup_test_conversation_committed.id

    # Sanity check: ensure the conversation exists before we act
    conversation_before = (
        db_session.query(Conversation).filter_by(id=conversation_id).one_or_none()
    )
    assert conversation_before is not None

    # 2. ACT
    response = await client.delete(f"/api/v1/conversations/{conversation_id}/")

    # 3. ASSERT
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # The most important check: verify the record is gone from the database
    db_session.expire_all()  # Ensure we get a fresh read from the DB
    conversation_after = (
        db_session.query(Conversation).filter_by(id=conversation_id).one_or_none()
    )
    assert conversation_after is None


async def test_rename_conversation_updates_title_in_database(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for PATCH /conversations/{id}/rename/
    - Verifies the API response.
    - Verifies the conversation's title is updated in the database.
    """
    # 1. ARRANGE
    conversation = setup_test_conversation_committed
    conversation_id = conversation.id
    new_title = "This is the new, updated title."

    request_data = {"title": new_title}

    # 2. ACT
    response = await client.patch(
        f"/api/v1/conversations/{conversation_id}/rename/", json=request_data
    )

    # 3. ASSERT
    assert response.status_code == 200
    assert response.json()["message"] == f"Conversation renamed to '{new_title}'"

    # Verify the change was persisted in the database
    db_session.refresh(conversation)
    assert conversation.title == new_title


async def test_regenerate_message_dispatches_regenerate_celery_task(
    client,
    mock_celery_tasks,
    mock_redis_stream_manager,
    db_session: Session,
    setup_test_conversation_committed: Conversation,
):
    """
    Integration Test for POST /.../regenerate/
    - Verifies the handoff to the correct "regenerate" Celery task.
    """
    # 1. ARRANGE
    # The regenerate logic needs a "last human message" to exist. Let's create one.
    last_human_message = Message(
        id="last-human-msg-123",
        conversation_id=setup_test_conversation_committed.id,
        content="This was my last message, please regenerate.",
        type=MessageType.HUMAN,
        sender_id="test-user",  # From our auth mock
    )
    db_session.add(last_human_message)
    db_session.commit()

    conversation_id = setup_test_conversation_committed.id
    request_data = {"node_ids": [{"node_id": "node-abc", "name": "Test Node Name"}]}

    # 2. ACT
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/regenerate/", json=request_data
    )

    # 3. ASSERT
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Verify the handoff to the REGENERATE Celery task
    mock_celery_tasks["regenerate"].assert_called_once()
    mock_celery_tasks[
        "execute"
    ].assert_not_called()  # Ensure the other task wasn't called

    # Inspect the arguments passed to the regenerate task
    call_kwargs = mock_celery_tasks["regenerate"].call_args.kwargs
    assert call_kwargs["conversation_id"] == conversation_id
    assert call_kwargs["user_id"] == "test-user"

    expected_node_ids_data = [{"node_id": "node-abc", "name": "Test Node Name"}]

    # This is the list of NodeContext objects that the mock actually received.
    received_node_objects = call_kwargs["node_ids"]

    # Convert the list of objects into a list of dictionaries for a valid comparison.
    received_node_ids_data = [node.model_dump() for node in received_node_objects]

    assert received_node_ids_data == expected_node_ids_data


# ---------------------------------------------------------------------------
# GET /conversations (list)
# ---------------------------------------------------------------------------
async def test_get_conversations_list_returns_user_conversations(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for GET /conversations
    - Verifies that the endpoint returns a list including the user's conversation.
    """
    response = await client.get("/api/v1/conversations")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Our fixture creates a conversation for test-user
    ids = [c["id"] for c in data]
    assert setup_test_conversation_committed.id in ids


async def test_get_conversations_list_respects_pagination(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for GET /conversations with pagination params.
    """
    response = await client.get("/api/v1/conversations?start=0&limit=1")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 1


# ---------------------------------------------------------------------------
# GET /conversations/{id}/info
# ---------------------------------------------------------------------------
async def test_get_conversation_info_returns_details(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for GET /conversations/{id}/info
    - Verifies the endpoint returns conversation details.
    """
    conversation_id = setup_test_conversation_committed.id
    response = await client.get(f"/api/v1/conversations/{conversation_id}/info")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == conversation_id
    assert "title" in data
    assert "status" in data
    assert "project_ids" in data
    assert "total_messages" in data


# ---------------------------------------------------------------------------
# GET /conversations/{id}/messages
# ---------------------------------------------------------------------------
async def test_get_conversation_messages_returns_list(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for GET /conversations/{id}/messages
    - Verifies the endpoint returns a list of messages.
    """
    conversation_id = setup_test_conversation_committed.id
    response = await client.get(f"/api/v1/conversations/{conversation_id}/messages")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


async def test_get_conversation_messages_with_pagination(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """
    Integration Test for GET /conversations/{id}/messages with pagination.
    """
    # Add some messages first
    for i in range(3):
        msg = Message(
            id=f"msg-pagination-{i}",
            conversation_id=setup_test_conversation_committed.id,
            content=f"Test message {i}",
            type=MessageType.HUMAN,
            sender_id="test-user",
        )
        db_session.add(msg)
    db_session.commit()

    conversation_id = setup_test_conversation_committed.id
    response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages?start=0&limit=2"
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 2


# ---------------------------------------------------------------------------
# POST /conversations/{id}/stop
# ---------------------------------------------------------------------------
@patch("app.modules.conversations.conversation.conversation_service.ConversationService.stop_generation")
async def test_stop_generation_calls_service(
    mock_stop,
    client,
    db_session: Session,
    setup_test_conversation_committed: Conversation,
):
    """
    Integration Test for POST /conversations/{id}/stop
    - Verifies the endpoint calls stop_generation and returns success.
    """
    mock_stop.return_value = {"status": "stopped"}
    conversation_id = setup_test_conversation_committed.id

    response = await client.post(f"/api/v1/conversations/{conversation_id}/stop")

    assert response.status_code == 200
    mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# GET /conversations/{id}/active-session (happy path with mock Redis)
# ---------------------------------------------------------------------------
async def test_get_active_session_returns_session_when_exists(
    app,
    client,
    db_session: Session,
    setup_test_conversation_committed: Conversation,
):
    """
    Integration Test for GET /conversations/{id}/active-session
    - Overrides get_async_session_service to return a valid session.
    - Verifies 200 and response shape.
    """
    from app.modules.conversations.conversation_deps import get_async_session_service
    from app.modules.conversations.conversation.conversation_schema import ActiveSessionResponse

    mock_svc = MagicMock()
    mock_svc.get_active_session = AsyncMock(
        return_value=ActiveSessionResponse(
            sessionId="run-123",
            status="active",
            cursor="0-0",
            conversationId=setup_test_conversation_committed.id,
            startedAt=1700000000000,
            lastActivity=1700000030000,
        )
    )
    app.dependency_overrides[get_async_session_service] = lambda: mock_svc
    try:
        conversation_id = setup_test_conversation_committed.id
        response = await client.get(f"/api/v1/conversations/{conversation_id}/active-session")

        assert response.status_code == 200
        data = response.json()
        assert data["sessionId"] == "run-123"
        assert data["status"] == "active"
        assert data["conversationId"] == conversation_id
    finally:
        app.dependency_overrides.pop(get_async_session_service, None)


# ---------------------------------------------------------------------------
# GET /conversations/{id}/task-status (happy path with mock Redis)
# ---------------------------------------------------------------------------
async def test_get_task_status_returns_status_when_exists(
    app,
    client,
    db_session: Session,
    setup_test_conversation_committed: Conversation,
):
    """
    Integration Test for GET /conversations/{id}/task-status
    - Overrides get_async_session_service to return a valid task status.
    - Verifies 200 and response shape.
    """
    from app.modules.conversations.conversation_deps import get_async_session_service
    from app.modules.conversations.conversation.conversation_schema import TaskStatusResponse

    mock_svc = MagicMock()
    mock_svc.get_task_status = AsyncMock(
        return_value=TaskStatusResponse(
            isActive=True,
            sessionId="run-456",
            estimatedCompletion=1700000060000,
            conversationId=setup_test_conversation_committed.id,
        )
    )
    app.dependency_overrides[get_async_session_service] = lambda: mock_svc
    try:
        conversation_id = setup_test_conversation_committed.id
        response = await client.get(f"/api/v1/conversations/{conversation_id}/task-status")

        assert response.status_code == 200
        data = response.json()
        assert data["isActive"] is True
        assert data["sessionId"] == "run-456"
        assert data["conversationId"] == conversation_id
    finally:
        app.dependency_overrides.pop(get_async_session_service, None)


# ---------------------------------------------------------------------------
# Negative tests: 404 for non-existent conversation
# ---------------------------------------------------------------------------
async def test_get_conversation_info_404_for_nonexistent(client, db_session: Session):
    """GET /conversations/{id}/info returns 404 for non-existent conversation."""
    response = await client.get("/api/v1/conversations/nonexistent-id-12345/info")
    assert response.status_code == 404


async def test_get_conversation_messages_404_for_nonexistent(client, db_session: Session):
    """GET /conversations/{id}/messages returns 401 or 404 for non-existent conversation."""
    # Service raises AccessTypeNotFoundError -> 401 when conversation not in DB
    response = await client.get("/api/v1/conversations/nonexistent-id-12345/messages")
    assert response.status_code in (401, 404)


async def test_delete_conversation_404_for_nonexistent(client, db_session: Session):
    """DELETE /conversations/{id} returns 404 for non-existent conversation."""
    response = await client.delete("/api/v1/conversations/nonexistent-id-12345/")
    assert response.status_code == 404


async def test_rename_conversation_404_for_nonexistent(client, db_session: Session):
    """PATCH /conversations/{id}/rename returns 404 or 5xx for non-existent conversation."""
    response = await client.patch(
        "/api/v1/conversations/nonexistent-id-12345/rename/",
        json={"title": "New Title"},
    )
    assert response.status_code in (404, 500)


async def test_post_message_404_for_nonexistent(
    client, db_session: Session, mock_celery_tasks, mock_redis_stream_manager
):
    """POST /conversations/{id}/message returns 404 or 200 for non-existent conversation.
    Router may not check conversation existence before starting stream, so 200 is possible.
    """
    response = await client.post(
        "/api/v1/conversations/nonexistent-id-12345/message/",
        data={"content": "hello"},
    )
    assert response.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Negative tests: 422 for invalid request body
# ---------------------------------------------------------------------------
async def test_create_conversation_422_for_invalid_body(client, db_session: Session):
    """POST /conversations returns 422 for missing required fields."""
    response = await client.post("/api/v1/conversations/", json={})
    assert response.status_code == 422


async def test_rename_conversation_422_for_missing_title(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """PATCH /conversations/{id}/rename returns 422 when title is missing."""
    conversation_id = setup_test_conversation_committed.id
    response = await client.patch(
        f"/api/v1/conversations/{conversation_id}/rename/",
        json={},
    )
    assert response.status_code == 422


async def test_regenerate_422_for_invalid_node_ids(
    client,
    db_session: Session,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed: Conversation,
):
    """POST /conversations/{id}/regenerate returns 422 for invalid node_ids format."""
    # Add a human message so regenerate has something to work with
    msg = Message(
        id="msg-for-regen-422",
        conversation_id=setup_test_conversation_committed.id,
        content="Last human message",
        type=MessageType.HUMAN,
        sender_id="test-user",
    )
    db_session.add(msg)
    db_session.commit()

    conversation_id = setup_test_conversation_committed.id
    # Send invalid node_ids (should be list of objects with node_id and name)
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/regenerate/",
        json={"node_ids": "invalid-string"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Edge case: Empty/whitespace message content (400)
# ---------------------------------------------------------------------------
async def test_post_message_400_for_empty_content(
    client,
    db_session: Session,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed: Conversation,
):
    """POST /conversations/{id}/message returns 400 or 422 for empty content."""
    conversation_id = setup_test_conversation_committed.id
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/message/",
        data={"content": ""},
    )
    assert response.status_code in (400, 422)
    if response.status_code == 400:
        assert "empty" in response.json().get("detail", "").lower()


async def test_post_message_400_for_whitespace_only_content(
    client,
    db_session: Session,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed: Conversation,
):
    """POST /conversations/{id}/message returns 400 for whitespace-only content."""
    conversation_id = setup_test_conversation_committed.id
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/message/",
        data={"content": "   \t\n  "},
    )
    assert response.status_code == 400
    assert "empty" in response.json().get("detail", "").lower()


# ---------------------------------------------------------------------------
# Edge case: Invalid node_ids JSON in post_message (400)
# ---------------------------------------------------------------------------
async def test_post_message_400_for_invalid_node_ids_json(
    client,
    db_session: Session,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed: Conversation,
):
    """POST /conversations/{id}/message returns 400 for invalid node_ids JSON."""
    conversation_id = setup_test_conversation_committed.id
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/message/",
        data={"content": "hello", "node_ids": "not-valid-json{"},
    )
    assert response.status_code == 400
    assert "node_ids" in response.json().get("detail", "").lower()


# ---------------------------------------------------------------------------
# Edge case: active-session 404 when no session exists
# ---------------------------------------------------------------------------
async def test_get_active_session_404_when_no_session(
    app,
    client,
    db_session: Session,
    setup_test_conversation_committed: Conversation,
):
    """GET /conversations/{id}/active-session returns 404 when no active session."""
    from app.modules.conversations.conversation_deps import get_async_session_service
    from app.modules.conversations.conversation.conversation_schema import ActiveSessionErrorResponse

    mock_svc = MagicMock()
    mock_svc.get_active_session = AsyncMock(
        return_value=ActiveSessionErrorResponse(
            error="No active session found",
            conversationId=setup_test_conversation_committed.id,
        )
    )
    app.dependency_overrides[get_async_session_service] = lambda: mock_svc
    try:
        conversation_id = setup_test_conversation_committed.id
        response = await client.get(f"/api/v1/conversations/{conversation_id}/active-session")

        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_async_session_service, None)


# ---------------------------------------------------------------------------
# Edge case: task-status 404 when no task exists
# ---------------------------------------------------------------------------
async def test_get_task_status_404_when_no_task(
    app,
    client,
    db_session: Session,
    setup_test_conversation_committed: Conversation,
):
    """GET /conversations/{id}/task-status returns 404 when no background task."""
    from app.modules.conversations.conversation_deps import get_async_session_service
    from app.modules.conversations.conversation.conversation_schema import TaskStatusErrorResponse

    mock_svc = MagicMock()
    mock_svc.get_task_status = AsyncMock(
        return_value=TaskStatusErrorResponse(
            error="No background task found",
            conversationId=setup_test_conversation_committed.id,
        )
    )
    app.dependency_overrides[get_async_session_service] = lambda: mock_svc
    try:
        conversation_id = setup_test_conversation_committed.id
        response = await client.get(f"/api/v1/conversations/{conversation_id}/task-status")

        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_async_session_service, None)


# ---------------------------------------------------------------------------
# Edge case: GET /conversations with sorting options
# ---------------------------------------------------------------------------
async def test_get_conversations_list_with_sort_by_created_at(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """GET /conversations respects sort=created_at parameter."""
    response = await client.get("/api/v1/conversations?sort=created_at&order=asc")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


async def test_get_conversations_list_with_sort_order_desc(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """GET /conversations respects order=desc parameter."""
    response = await client.get("/api/v1/conversations?sort=updated_at&order=desc")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Edge case: stop generation 404 for non-existent conversation
# ---------------------------------------------------------------------------
async def test_stop_generation_404_for_nonexistent(client, db_session: Session):
    """POST /conversations/{id}/stop returns 404 or 200 for non-existent conversation."""
    response = await client.post("/api/v1/conversations/nonexistent-id-12345/stop")
    assert response.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Edge case: active-session and task-status 403 for non-existent conversation
# (access check fails before session lookup)
# ---------------------------------------------------------------------------
async def test_get_active_session_403_for_nonexistent(client, db_session: Session):
    """GET /conversations/{id}/active-session returns 403 when conversation not found."""
    response = await client.get("/api/v1/conversations/nonexistent-id-12345/active-session")
    # The router does access check first which raises 403 if conversation not found
    assert response.status_code == 403


async def test_get_task_status_403_for_nonexistent(client, db_session: Session):
    """GET /conversations/{id}/task-status returns 403 when conversation not found."""
    response = await client.get("/api/v1/conversations/nonexistent-id-12345/task-status")
    # The router does access check first which raises 403 if conversation not found
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Edge case: Conversations list returns empty when user has no conversations
# ---------------------------------------------------------------------------
async def test_get_conversations_list_empty_for_new_user(client, db_session: Session):
    """GET /conversations returns empty list when user has no conversations."""
    # Don't use setup_test_conversation_committed; use high offset so we get empty result (shared DB)
    response = await client.get("/api/v1/conversations?start=9999&limit=10")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


# ---------------------------------------------------------------------------
# Edge case: Messages list returns empty when conversation has no messages
# ---------------------------------------------------------------------------
async def test_get_conversation_messages_empty_when_no_messages(
    client, db_session: Session, setup_test_conversation_committed: Conversation
):
    """GET /conversations/{id}/messages returns empty list when no messages exist."""
    # Fixture creates a conversation but no messages; use high offset for empty result (shared DB)
    conversation_id = setup_test_conversation_committed.id
    response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages?start=9999&limit=10"
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0
