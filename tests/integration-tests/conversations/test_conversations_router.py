import pytest
from unittest.mock import patch, AsyncMock, MagicMock, ANY
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType
from app.modules.media.media_schema import AttachmentUploadResponse
from app.modules.users.user_model import User
from sqlalchemy.orm import Session
from app.modules.projects.projects_model import Project


pytestmark = pytest.mark.asyncio


# THIS IS THE FIX: We are patching the name 'MediaService' exactly where it is used.
@patch("app.modules.conversations.conversations_router.MediaService")
async def test_post_message_successful_flow(
    mock_media_service_class, client, db_session
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
    ) as mock_celery_task, patch(
        "app.modules.conversations.utils.redis_streaming.RedisStreamManager.wait_for_task_start",
        return_value=True,
    ):

        # Configure the mock INSTANCE's methods. The __init__ is now completely skipped.
        mock_instance.upload_image = AsyncMock(
            return_value=AttachmentUploadResponse(
                id="fake_attachment_id",
                attachment_type="image",
                file_name="test_image.png",
                mime_type="image/png",
                file_size=1000,
            )
        )

        # Prepare request data
        conversation_id = "some_conversation_id"
        files = {"images": ("test_image.png", b"fake image data", "image/png")}
        data = {"content": "This is a test message."}

        # Make the HTTP request
        response = await client.post(
            f"/api/v1/conversations/{conversation_id}/message/", files=files, data=data
        )

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
    assert len(db_messages) == 1
    assert (
        db_messages[0].content
        == "You can now ask questions about the Test Project Repo repository."
    )


async def test_post_message_dispatches_celery_task_and_streams(
    client,
    mock_celery_tasks,
    mock_redis_stream_manager,
    setup_test_conversation_committed: Conversation,  # Depend on all our data fixtures
):
    """
    Integration Test for POST /.../message/
    - Verifies the API response is a streaming response.
    - Verifies the RedisStreamManager is called to prepare the stream.
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
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/message/", data=form_data
    )

    # 3. ASSERT

    # Part A: Verify the API response
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Part B: Verify the interaction with Redis
    # Check that our code tried to wait for the background task to start
    mock_redis_stream_manager.wait_for_task_start.assert_called_once()

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
