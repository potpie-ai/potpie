"""In-memory persistent watch expression store.

Keyed by (user_id, conversation_id) so each debug session has its own watch list.
TTL is implicit — the process lifetime covers the chat session.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, List, Optional, Tuple

_store: Dict[Tuple[Optional[str], Optional[str]], List[str]] = {}
_lock = Lock()


def _key(user_id: Optional[str], conversation_id: Optional[str]) -> Tuple:
    return (user_id, conversation_id)


def add_watch(user_id: Optional[str], conversation_id: Optional[str], expression: str) -> List[str]:
    k = _key(user_id, conversation_id)
    with _lock:
        watches = _store.setdefault(k, [])
        if expression not in watches:
            watches.append(expression)
        return list(watches)


def remove_watch(user_id: Optional[str], conversation_id: Optional[str], expression: str) -> List[str]:
    k = _key(user_id, conversation_id)
    with _lock:
        watches = _store.get(k, [])
        _store[k] = [w for w in watches if w != expression]
        return list(_store[k])


def list_watches(user_id: Optional[str], conversation_id: Optional[str]) -> List[str]:
    k = _key(user_id, conversation_id)
    with _lock:
        return list(_store.get(k, []))


def clear_watches(user_id: Optional[str], conversation_id: Optional[str]) -> None:
    k = _key(user_id, conversation_id)
    with _lock:
        _store.pop(k, None)


def merge_into_expressions(
    user_id: Optional[str],
    conversation_id: Optional[str],
    expressions: Optional[List[str]],
) -> List[str]:
    """Return the union of persistent watches and the caller-supplied expressions, deduped."""
    watches = list_watches(user_id, conversation_id)
    combined = list(watches)
    for expr in expressions or []:
        if expr not in combined:
            combined.append(expr)
    return combined
