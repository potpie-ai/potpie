import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app  # Assuming the FastAPI app is initialized in app.main
from app.core.database import Base, get_db
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType, MessageStatus

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the database dependency to use the test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="function")
def db():
    """Set up a clean database for each test."""
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

# Mock authenticated user (assuming AuthService.check_auth returns user information)
@pytest.fixture
def mock_user():
    return {"user_id": "test_user_id", "email": "test@example.com"}

# Helper function to add a conversation to the database
def create_test_conversation(db, conversation_id, title, user_id):
    conversation = Conversation(
        id=conversation_id,
        user_id=user_id,
        title=title,
        status="active"
    )
    db.add(conversation)
    db.commit()
    return conversation

# Helper function to add a message to the conversation
def create_test_message(db, conversation_id, content, message_type, user_id):
    message = Message(
        id="msg_1",
        conversation_id=conversation_id,
        content=content,
        sender_id=user_id,
        type=message_type,
        status=MessageStatus.ACTIVE
    )
    db.add(message)
    db.commit()
    return message


# Test for creating a conversation (POST /conversations/)
@pytest.mark.asyncio
async def test_create_conversation(db, mock_user):
    # Prepare the input for creating a conversation
    conversation_data = {
        "title": "New Conversation",
        "project_ids": ["project_1"],
        "agent_ids": ["debugging_agent"]
    }

    # Simulate a request to create a conversation
    response = client.post("/conversations/", json=conversation_data, headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert "conversation_id" in response_data
    assert response_data["message"] == "Conversation created successfully."

    # Validate that the conversation was created in the database
    conversation = db.query(Conversation).filter_by(id=response_data["conversation_id"]).first()
    assert conversation is not None
    assert conversation.title == "New Conversation"

# Test for fetching conversation info (GET /conversations/{conversation_id}/info/)
@pytest.mark.asyncio
async def test_get_conversation_info(db, mock_user):
    # Pre-create a conversation in the database
    conversation = create_test_conversation(db, "conversation_1", "Test Conversation", "test_user_id")

    # Simulate a request to fetch the conversation info
    response = client.get(f"/conversations/{conversation.id}/info/", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["id"] == conversation.id
    assert response_data["title"] == conversation.title
    assert response_data["status"] == conversation.status

# Test for posting a message (POST /conversations/{conversation_id}/message/)
@pytest.mark.asyncio
async def test_post_message(db, mock_user):
    # Pre-create a conversation in the database
    conversation = create_test_conversation(db, "conversation_1", "Test Conversation", "test_user_id")

    # Prepare message request data
    message_data = {"content": "Hello, how are you?"}

    # Simulate a request to post a message
    response = client.post(f"/conversations/{conversation.id}/message/", json=message_data, headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200

    # Check if the message was stored in the database
    message = db.query(Message).filter_by(conversation_id=conversation.id).first()
    assert message is not None
    assert message.content == "Hello, how are you?"
    assert message.type == MessageType.HUMAN

# Test for deleting a conversation (DELETE /conversations/{conversation_id}/)
@pytest.mark.asyncio
async def test_delete_conversation(db, mock_user):
    # Pre-create a conversation in the database
    conversation = create_test_conversation(db, "conversation_1", "Test Conversation", "test_user_id")

    # Simulate a request to delete the conversation
    response = client.delete(f"/conversations/{conversation.id}/", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["message"] == f"Conversation {conversation.id} and its messages have been permanently deleted."

    # Check if the conversation is deleted from the database
    deleted_conversation = db.query(Conversation).filter_by(id="conversation_1").first()
    assert deleted_conversation is None

# Test for renaming a conversation (PATCH /conversations/{conversation_id}/rename/)
@pytest.mark.asyncio
async def test_rename_conversation(db, mock_user):
    # Pre-create a conversation in the database
    conversation = create_test_conversation(db, "conversation_1", "Old Title", "test_user_id")

    # Prepare rename request data
    rename_data = {"title": "New Title"}

    # Simulate a request to rename the conversation
    response = client.patch(f"/conversations/{conversation.id}/rename/", json=rename_data, headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["message"] == "Conversation renamed to 'New Title'"

    # Check if the conversation title is updated in the database
    updated_conversation = db.query(Conversation).filter_by(id="conversation_1").first()
    assert updated_conversation.title == "New Title"

# Test for regenerating the last message (POST /conversations/{conversation_id}/regenerate/)
@pytest.mark.asyncio
async def test_regenerate_last_message(db, mock_user):
    # Pre-create a conversation and message in the database
    conversation = create_test_conversation(db, "conversation_1", "Test Conversation", "test_user_id")
    create_test_message(db, conversation.id, "Last message content", MessageType.HUMAN, "test_user_id")

    # Simulate a request to regenerate the last message
    response = client.post(f"/conversations/{conversation.id}/regenerate/", headers={"Authorization": "Bearer test_token"})

    # Validate response
    assert response.status_code == 200

    # You can further extend the validation based on your response and use case
