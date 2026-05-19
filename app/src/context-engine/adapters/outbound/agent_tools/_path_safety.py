"""Boundary validators for agent-supplied git refs and worktree paths.

The reconciliation agent is prompt-injectable, so every ``ref``/``path``
it hands to a sandbox tool is untrusted. These checks run at the adapter
boundary — independent of whatever the sandbox engine enforces — to block
git option-injection (a ref/arg parsed as a ``--flag``) and path traversal
out of the pot's worktree (security review H-4 / H-5).
"""

from __future__ import annotations

import re

# First char must be alnum so a value can never begin with ``-`` (option
# injection). Body excludes ``:`` (refspec / ``ext::`` transport),
# whitespace, backslash and shell-ish metacharacters. Covers normal refs:
# SHAs, ``HEAD``, ``HEAD~3``, ``HEAD^``, ``origin/main``, ``v1.2.3``,
# ``feature/x``, ``HEAD@{1}``.
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+~^@{}\-]*$")


def is_safe_git_ref(value: str | None) -> bool:
    """True when ``value`` is a safe git revision/refname argument."""
    if not value or len(value) > 256:
        return False
    if value[0] == "-":  # redundant with the regex; explicit + cheap
        return False
    if ".." in value:  # range/parent-walk belongs in caller-built args
        return False
    return bool(_GIT_REF_RE.match(value))


def is_safe_date_expr(value: str | None) -> bool:
    """``--since=`` value: a single token, no control chars / newlines."""
    if value is None:
        return True
    if not value or len(value) > 128:
        return False
    if value[0] == "-":
        return False
    return all(ch >= " " and ch != "\x7f" for ch in value)


def is_safe_relpath(value: str | None) -> bool:
    """True when ``value`` is a worktree-relative path with no escape.

    Rejects empty, absolute (POSIX or Windows-drive), any ``..`` segment,
    NUL/control bytes. ``.`` is allowed (``sandbox_list_dir`` default).
    """
    if value is None or value == "":
        return False
    if "\x00" in value or any(ch < " " for ch in value):
        return False
    if value.startswith("/") or value.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", value):  # C:\ , D:/ ...
        return False
    norm = value.replace("\\", "/")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    return ".." not in parts
