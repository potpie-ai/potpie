import pytest

from app.modules.intelligence.agents.chat_agent import ChatContext


pytestmark = pytest.mark.unit


def test_chat_context_image_helpers_merge_and_preserve_original_data():
    current = {"img-1": {"mime_type": "image/png", "file_size": 10}}
    history = {"img-2": {"mime_type": "image/jpeg", "file_size": 20}}
    ctx = ChatContext(
        project_id="project-1",
        project_name="Project",
        curr_agent_id="agent-1",
        history=["hello"],
        query="find issue",
        image_attachments=current,
        context_images=history,
    )

    all_images = ctx.get_all_images()

    assert ctx.has_images() is True
    assert all_images["img-1"]["context_type"] == "current_message"
    assert all_images["img-1"]["relevance"] == "high"
    assert all_images["img-2"]["context_type"] == "conversation_history"
    assert all_images["img-2"]["relevance"] == "medium"
    assert current["img-1"] == {"mime_type": "image/png", "file_size": 10}
    assert history["img-2"] == {"mime_type": "image/jpeg", "file_size": 20}


def test_chat_context_helpers_return_empty_defaults_without_images():
    ctx = ChatContext(
        project_id="project-2",
        project_name="Project",
        curr_agent_id="agent-2",
        history=[],
        query="status",
    )

    assert ctx.is_inferring() is False
    assert ctx.has_images() is False
    assert ctx.get_all_images() == {}
    assert ctx.get_current_images_only() == {}
    assert ctx.get_context_images_only() == {}


def test_chat_context_is_inferring_for_inferring_projects():
    ctx = ChatContext(
        project_id="project-3",
        project_name="Project",
        curr_agent_id="agent-3",
        history=[],
        query="status",
        project_status="inferring",
    )

    assert ctx.is_inferring() is True
