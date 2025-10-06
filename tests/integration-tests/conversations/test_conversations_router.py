import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType 
from app.modules.media.media_schema import AttachmentUploadResponse
from sqlalchemy.orm import Session

pytestmark = pytest.mark.asyncio

# THIS IS THE FIX: We are patching the name 'MediaService' exactly where it is used.
@patch('app.modules.conversations.conversations_router.MediaService')
async def test_post_message_successful_flow(mock_media_service_class, client, db_session):
    """
    Tests the complete, successful flow of a user posting a new message
    with an image.
    """
    # Get the mock INSTANCE that is created when the router code calls MediaService(db).
    mock_instance = mock_media_service_class.return_value

    # Patch other external dependencies
    with patch('app.celery.tasks.agent_tasks.execute_agent_background.delay') as mock_celery_task, \
         patch('app.modules.conversations.utils.redis_streaming.RedisStreamManager.wait_for_task_start', return_value=True):

        # Configure the mock INSTANCE's methods. The __init__ is now completely skipped.
        mock_instance.upload_image = AsyncMock(
            return_value=AttachmentUploadResponse(
                id="fake_attachment_id",
                attachment_type="image",
                file_name="test_image.png",
                mime_type="image/png",
                file_size=1000
            )
        )

        # Prepare request data
        conversation_id = "some_conversation_id"
        files = {'images': ('test_image.png', b"fake image data", 'image/png')}
        data = {'content': "This is a test message."}

        # Make the HTTP request
        response = await client.post(
            f"/api/v1/conversations/{conversation_id}/message/",
            files=files,
            data=data
        )

        # Assert outcomes
        assert response.status_code == 200
        assert 'text/event-stream' in response.headers['content-type']
        
        # Assert calls on the mock INSTANCE
        mock_instance.upload_image.assert_called_once()
        mock_celery_task.assert_called_once()
        

@patch('app.modules.conversations.conversation.conversation_service.InferenceService') # <-- CRITICAL: Patch where InferenceService is USED
async def test_create_conversation_successful_flow(
    mock_inference_service_class: MagicMock, 
    client, 
    db_session: Session
):
    """
    Tests the complete, successful flow of creating a new conversation,
    with assertions updated to match the precise database schema.
    """
    # =================================================================
    # 1. Setup ("Arrange" Phase)
    # =================================================================
    
    # Mock the InferenceService to return a predictable, fake response.
    mock_inference_instance = mock_inference_service_class.return_value
    mock_inference_instance.get_response = AsyncMock(return_value="Hello! I am a mock AI.")

    # The request payload from the user
    request_data = {
        "message": "Hello, world!"
    }

    # =================================================================
    # 2. Execution ("Act" Phase)
    # =================================================================

    # Make a POST request to the create conversation endpoint.
    response = await client.post(
        "/api/v1/conversations", # Adjust if your API prefix is different
        json=request_data
    )

    # =================================================================
    # 3. Verification ("Assert" Phase)
    # =================================================================

    # Part A: Assert the API Response
    assert response.status_code == 201, f"Expected 201 Created, but got {response.status_code}: {response.text}"
    
    response_data = response.json()
    assert "id" in response_data
    assert response_data["initial_message"] == "Hello, world!" # This assumes your API returns this field. Adjust if needed.
    
    conversation_id = response_data["id"]

    # Part B: Assert the Mocked Service Calls
    mock_inference_instance.get_response.assert_called_once()

    # Part C: Assert the Database State (Updated for the Message model)
    # Fetch the conversation to ensure it was created correctly.
    new_conversation = db_session.query(Conversation).filter_by(id=conversation_id).one_or_none()
    assert new_conversation is not None
    assert new_conversation.user_id == "test-user" # This comes from the auth mock in conftest.py

    # Fetch the messages associated with this conversation.
    messages = db_session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()
    
    assert len(messages) == 2, "Expected two messages in the database: the user's and the AI's."
    
    # Verify the first message (from the user) based on your schema
    user_message = messages[0]
    assert user_message.content == "Hello, world!"
    assert user_message.type == MessageType.HUMAN # <-- CHANGED: Uses the MessageType Enum
    assert user_message.sender_id == "test-user"   # <-- ADDED: Verifies the sender_id for HUMAN messages
    
    # Verify the second message (from the mocked AI) based on your schema
    agent_message = messages[1]
    assert agent_message.content == "Hello! I am a mock AI."
    assert agent_message.type == MessageType.AI_GENERATED # <-- CHANGED: Uses the MessageType Enum
    assert agent_message.sender_id is None  