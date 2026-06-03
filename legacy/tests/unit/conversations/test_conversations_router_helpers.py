import pytest

from app.modules.conversations.conversations_router import (
    _is_vscode_extension_user_agent,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("Potpie-VSCode-Extension/1.0", True),
        ("Potpie-VSCode-Extension/1.0.0", True),
        ("Mozilla Potpie-VSCode-Extension/12.34 tail", True),
        ("Potpie-VSCode-Extension", False),
        ("Potpie-VSCode-Extension/abc", False),
        ("NotPotpie-VSCode-Extension/1.0", False),
        ("Potpie-VSCode-Extension/1", False),
    ],
)
def test_is_vscode_extension_user_agent(user_agent, expected):
    assert _is_vscode_extension_user_agent(user_agent) is expected
