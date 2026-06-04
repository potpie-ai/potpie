"""H-4 / H-5: boundary validators for agent-supplied git refs + paths."""

from __future__ import annotations

import pytest

from adapters.outbound.agent_tools._path_safety import (
    is_safe_date_expr,
    is_safe_git_ref,
    is_safe_relpath,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "ref",
    [
        "main",
        "origin/main",
        "feature/foo-bar",
        "v1.2.3",
        "HEAD",
        "HEAD~3",
        "HEAD^",
        "a1b2c3d4e5f6",
        "release/2026.05",
    ],
)
def test_safe_refs_accepted(ref):
    assert is_safe_git_ref(ref) is True


@pytest.mark.parametrize(
    "ref",
    [
        "",
        None,
        "-rf",  # option injection
        "--upload-pack=touch /tmp/x",
        "--output=/etc/cron.d/x",
        "ext::sh -c whoami",  # transport via leading 'ext::' → ':' barred
        "a b",  # whitespace
        "a..b",  # range belongs in caller-built args
        "branch;rm -rf /",
        "x\nname",
        "a" * 300,
    ],
)
def test_unsafe_refs_rejected(ref):
    assert is_safe_git_ref(ref) is False


@pytest.mark.parametrize("p", [".", "src", "src/app/main.py", "a/b/c.txt"])
def test_safe_relpaths_accepted(p):
    assert is_safe_relpath(p) is True


@pytest.mark.parametrize(
    "p",
    [
        "",
        None,
        "/etc/shadow",
        "../../etc/passwd",
        "src/../../secret",
        "a/../../b",
        "..",
        "C:\\Windows\\system32",
        "\\\\server\\share",
        "foo\x00bar",
    ],
)
def test_unsafe_relpaths_rejected(p):
    assert is_safe_relpath(p) is False


def test_date_expr_validation():
    assert is_safe_date_expr(None) is True
    assert is_safe_date_expr("2 weeks ago") is True
    assert is_safe_date_expr("2026-01-01") is True
    assert is_safe_date_expr("") is False
    assert is_safe_date_expr("-bad") is False
    assert is_safe_date_expr("a\nb") is False
