import pytest

from app.modules.intelligence.tools import tool_utils


pytestmark = pytest.mark.unit


def test_truncate_response_returns_original_when_short():
    assert tool_utils.truncate_response("short", max_length=10) == "short"


def test_truncate_response_appends_notice_when_content_is_long():
    content = "abcdefghijXYZ"

    truncated = tool_utils.truncate_response(content, max_length=10)

    assert truncated.startswith("abcdefghij")
    assert "[TRUNCATED]" in truncated
    assert "13 total characters" in truncated


def test_truncate_dict_response_truncates_nested_strings_and_lists():
    long_value = "x" * 12
    response = {
        "text": long_value,
        "nested": {"inner": long_value},
        "items": [long_value, {"deep": long_value}, 7],
        "count": 3,
    }

    truncated = tool_utils.truncate_dict_response(response, max_length=10)

    assert truncated["text"].startswith("x" * 10)
    assert "[TRUNCATED]" in truncated["text"]
    assert truncated["nested"]["inner"].startswith("x" * 10)
    assert truncated["items"][0].startswith("x" * 10)
    assert truncated["items"][1]["deep"].startswith("x" * 10)
    assert truncated["items"][2] == 7
    assert truncated["count"] == 3
