"""Redactor for recorded webhook payloads.

The redactor is conservative: it strips fields it knows are sensitive
and rewrites identities through a pinned alias map. New fixtures should
pass through ``redact_envelope()`` before being committed.

The redaction policy lives entirely in this file so reviewers can audit
it in one place.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TOKEN_RE = re.compile(
    r"(?i)(token|secret|api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"
)

# Stable alias map: when capturing fixtures, real usernames get rewritten
# to one of these. Add more as needed; do not remove existing entries
# (would change idempotency keys of historical fixtures).
USER_ALIAS_MAP = {
    # real_username : alias
}

DEFAULT_REPO_ALIAS = "acme/sandbox"

REDACTED = "[REDACTED]"
REDACTED_EMAIL_TEMPLATE = "{alias}@example.test"

SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
        "ssh_key",
        "installation_token",
        "X-Hub-Signature",
        "X-Hub-Signature-256",
        "Authorization",
    }
)


def _redact_email_in_string(s: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        local = match.group(0).split("@")[0]
        return REDACTED_EMAIL_TEMPLATE.format(alias=local[:8] or "user")

    return EMAIL_RE.sub(_replace, s)


def _strip_tokens(s: str) -> str:
    return TOKEN_RE.sub(lambda m: f"{m.group(1)}: {REDACTED}", s)


def _redact_value(value: Any, repo_alias: str) -> Any:
    if isinstance(value, str):
        v = _strip_tokens(_redact_email_in_string(value))
        # Only rewrite usernames we have an explicit alias for. Repo
        # rewriting must be opt-in per call, since "owner/repo" strings
        # in the wild are ambiguous.
        for real, alias in USER_ALIAS_MAP.items():
            v = v.replace(real, alias)
        return v
    if isinstance(value, dict):
        return _redact_dict(value, repo_alias)
    if isinstance(value, list):
        return [_redact_value(v, repo_alias) for v in value]
    return value


def _redact_dict(d: dict[str, Any], repo_alias: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        if key in SENSITIVE_KEYS:
            out[key] = REDACTED
            continue
        out[key] = _redact_value(value, repo_alias)
    return out


def redact_envelope(
    envelope: dict[str, Any], *, repo_alias: str = DEFAULT_REPO_ALIAS
) -> dict[str, Any]:
    """Return a deep-copied, redacted version of a fixture envelope.

    Records the redaction list under ``_meta.redactions`` so reviewers
    can confirm the fixture has been processed.
    """
    redacted = _redact_dict(deepcopy(envelope), repo_alias)
    meta = dict(redacted.get("_meta") or {})
    existing = list(meta.get("redactions") or [])
    for marker in ("emails", "tokens", "sensitive_keys"):
        if marker not in existing:
            existing.append(marker)
    meta["redactions"] = existing
    redacted["_meta"] = meta
    return redacted


def redact_file(path: Path, *, repo_alias: str = DEFAULT_REPO_ALIAS) -> None:
    """Read a fixture file, redact it in-place, and rewrite."""
    with path.open("r", encoding="utf-8") as f:
        envelope = json.load(f)
    redacted = redact_envelope(envelope, repo_alias=repo_alias)
    with path.open("w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=2, ensure_ascii=False)
        f.write("\n")
