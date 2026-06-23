"""Injection-ledger implementations (Graph V1.5 Step 12a).

``LocalInjectionLedger`` persists per-session injected keys to a JSON file under
the Potpie home (``$CONTEXT_ENGINE_HOME`` / ``~/.potpie``) so dedup survives
across the separate hook processes that fire within one harness session.
``InMemoryInjectionLedger`` is the dependency-free double for tests.

The store is read-modify-write per call (each hook is its own short-lived
process); concurrent hooks within a session may race on a write, but the cost of
a lost dedup entry is at worst one duplicate injection — never corruption — so a
lock is not warranted at this tier.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from context_engine.adapters.outbound.pots.local_pot_store import default_home

logger = logging.getLogger(__name__)

# Keep the file bounded: retain at most this many most-recently-touched sessions.
_MAX_SESSIONS = 500


class InMemoryInjectionLedger:
    """Process-local injection ledger (tests / ephemeral runs)."""

    def __init__(self) -> None:
        self._by_session: dict[str, set[str]] = {}

    def was_injected(self, session_id: str, key: str) -> bool:
        return key in self._by_session.get(session_id, set())

    def record(self, session_id: str, keys: Sequence[str]) -> None:
        bucket = self._by_session.setdefault(session_id, set())
        bucket.update(k for k in keys if k)


class LocalInjectionLedger:
    """JSON-file injection ledger under the Potpie home directory."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (default_home() / "nudge_sessions.json")

    # -- persistence ----------------------------------------------------------
    def _load(self) -> dict[str, dict]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:  # unreadable file → behave as empty, don't crash a hook
            logger.warning("injection ledger unreadable at %s: %s", self._path, exc)
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("injection ledger corrupt at %s; resetting", self._path)
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict]) -> None:
        # Prune to the most-recently-updated sessions to bound the file.
        if len(data) > _MAX_SESSIONS:
            ordered = sorted(
                data.items(),
                key=lambda kv: kv[1].get("updated_at", ""),
                reverse=True,
            )
            data = dict(ordered[:_MAX_SESSIONS])
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:  # a write failure must not break the hook
            logger.warning("could not persist injection ledger %s: %s", self._path, exc)

    # -- port -----------------------------------------------------------------
    def was_injected(self, session_id: str, key: str) -> bool:
        entry = self._load().get(session_id)
        if not entry:
            return False
        return key in set(entry.get("keys", ()))

    def record(self, session_id: str, keys: Sequence[str]) -> None:
        fresh = [k for k in keys if k]
        if not fresh:
            return
        data = self._load()
        entry = data.get(session_id) or {"keys": []}
        merged = list(dict.fromkeys([*entry.get("keys", []), *fresh]))
        data[session_id] = {
            "keys": merged,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save(data)


__all__ = ["InMemoryInjectionLedger", "LocalInjectionLedger"]
