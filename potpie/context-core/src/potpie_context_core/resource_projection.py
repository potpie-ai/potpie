"""Agent-visible projection for structured resource metadata.

Stored source bytes remain immutable. Key/value metadata that looks like a
credential is suppressed before search indexing or agent-visible retrieval.
Ordinary prose (including security documentation) is left untouched.
"""

from __future__ import annotations

import re
import json
from dataclasses import replace
from typing import Any, Mapping

from .ports.resource_store import Chunk

_FIELD = re.compile(
    r'(?m)(?P<prefix>(?:^|(?<=[{,]))\s*"(?P<key>[^"\\\r\n]{1,120})"\s*:\s*)'
    r'(?P<value>"(?:\\.|[^"\\])*"|[^,\r\n}]+)'
)
_SENSITIVE = re.compile(
    r"(?:^|[_-])(?:token|password|passwd|secret|cookie|api[_-]?key|private[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)


def project_resource_text(text: str) -> str:
    """Redact values of credential-like JSON fields, including JSON fragments."""
    if '"' not in text or ':' not in text:
        return text
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        projected = project_public_metadata(parsed)
        if projected != parsed:
            return json.dumps(projected, ensure_ascii=False)

    def replace_field(match: re.Match[str]) -> str:
        if not _credential_key(match.group("key")):
            return match.group(0)
        return match.group("prefix") + '"[redacted]"'

    return _FIELD.sub(replace_field, text)


def _credential_key(key: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return bool(_SENSITIVE.search(normalized))


def project_public_metadata(value: Any) -> Any:
    """Project structured metadata before an agent or diagnostic sees it."""
    if isinstance(value, str):
        return project_resource_text(value)
    if isinstance(value, Mapping):
        return {
            key: "[redacted]" if _credential_key(str(key)) else project_public_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [project_public_metadata(item) for item in value]
    return value


def project_chunk(chunk: Chunk) -> Chunk:
    safe_text = project_resource_text(chunk.text)
    return chunk if safe_text == chunk.text else replace(
        chunk, text=safe_text, chars=len(safe_text),
    )
