"""Local JSON-file pot store — the POC control-plane persistence.

Backs ``LocalPotManagementService`` so the active pot, pot list, and source
registry survive across CLI invocations (each ``potpie`` call is a fresh
process). State lives at ``<home>/pots.json`` where ``<home>`` is
``$CONTEXT_ENGINE_HOME`` or ``~/.potpie``.

This is intentionally a flat-file POC. The real control plane is the local
state DB (SQLite + migrations) per ``cli-flow.md``.

    TODO(stage-N): replace with the local state DB + migrations.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie.config.local_paths import default_home
from potpie.config.local_state import local_json_transaction


def _empty_state() -> dict[str, Any]:
    return {"pots": {}, "active": None, "sources": {}, "repo_defaults": {}}


@dataclass(slots=True)
class LocalPotStore:
    """Flat-file persistence for pots + sources + the active-pot pointer."""

    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "pots.json"

    # --- raw state ----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return _empty_state()

    # --- pots ---------------------------------------------------------------
    def list_pots(self) -> list[dict[str, Any]]:
        state = self._load()
        active = state.get("active")
        return [
            {**row, "active": pid == active}
            for pid, row in state.get("pots", {}).items()
        ]

    def active(self) -> dict[str, Any] | None:
        state = self._load()
        active = state.get("active")
        if not active:
            return None
        row = state.get("pots", {}).get(active)
        return {**row, "active": True} if row else None

    def create(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> dict[str, Any]:
        """Create a pot. Repo registration belongs on ``add_source`` via the CLI."""
        _ = repo
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            # Reuse an existing pot by name (idempotent setup), but never an
            # archived one: reuse would hand back a pot whose graph state was
            # cleared, under a name the caller expected to be fresh, and quietly
            # make it active again.
            for pid, row in state.get("pots", {}).items():
                if row.get("name") == name and not row.get("archived"):
                    if use:
                        state["active"] = pid
                    return {
                        **row,
                        "active": state.get("active") == pid,
                        "created": False,
                    }
            pot_id = f"pot_{uuid.uuid4().hex[:12]}"
            row = {"pot_id": pot_id, "name": name, "archived": False}
            state.setdefault("pots", {})[pot_id] = row
            if use or state.get("active") is None:
                state["active"] = pot_id
            return {**row, "active": state.get("active") == pot_id, "created": True}

    def _resolve_ref(
        self, state: dict[str, Any], ref: str, *, include_archived: bool = False
    ) -> str | None:
        """The pot id a ref names; a live pot always wins over an archived one.

        Archived pots are excluded unless ``include_archived``: they cannot be
        selected, written or routed to, so leaving them in the id/name space
        would let a retired pot shadow a live one that reuses its name. The
        lookup that decides which refusal to raise opts in to see them.
        """
        pots = state.get("pots", {})
        live = [pid for pid, row in pots.items() if not row.get("archived")]
        archived = [pid for pid, row in pots.items() if row.get("archived")]
        for candidates in (live, archived if include_archived else []):
            if ref in candidates:
                return ref
            for pid in candidates:
                if pots[pid].get("name") == ref:
                    return pid
        return None

    def find(self, *, ref: str, include_archived: bool = True) -> dict[str, Any] | None:
        """The row a ref names, without selecting or changing anything."""
        state = self._load()
        pid = self._resolve_ref(state, ref, include_archived=include_archived)
        if pid is None:
            return None
        return {**state["pots"][pid], "active": state.get("active") == pid}

    def names_in_use(self) -> dict[str, str]:
        """``name -> pot_id`` for every live pot, for uniqueness checks."""
        return {
            str(row.get("name")): pid
            for pid, row in self._load().get("pots", {}).items()
            if not row.get("archived") and row.get("name")
        }

    def pot_ids(self) -> frozenset[str]:
        """Every pot id, archived ones included (ids are never reused)."""
        return frozenset(self._load().get("pots", {}))

    def use(self, *, ref: str) -> dict[str, Any] | None:
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            pid = self._resolve_ref(state, ref)
            if pid is None:
                return None
            state["active"] = pid
            return {**state["pots"][pid], "active": True}

    def rename(self, *, ref: str, new_name: str) -> dict[str, Any] | None:
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            pid = self._resolve_ref(state, ref)
            if pid is None:
                return None
            state["pots"][pid]["name"] = new_name
            return {**state["pots"][pid], "active": state.get("active") == pid}

    def archive(self, *, ref: str) -> dict[str, Any] | None:
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            pid = self._resolve_ref(state, ref)
            if pid is None:
                return None
            state["pots"][pid]["archived"] = True
            if state.get("active") == pid:
                state["active"] = None
            return {**state["pots"][pid], "active": False}

    # --- sources ------------------------------------------------------------
    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> dict[str, Any]:
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            row = {
                "source_id": f"src_{uuid.uuid4().hex[:8]}",
                "kind": kind,
                "name": name or location,
                "location": location,
            }
            state.setdefault("sources", {}).setdefault(pot_id, []).append(row)
            return row

    def list_sources(self, *, pot_id: str) -> list[dict[str, Any]]:
        return self._load().get("sources", {}).get(pot_id, [])

    def list_repo_sources(self) -> list[dict[str, Any]]:
        """Repo sources of every live pot, joined to their pot, from one load.

        Pot order follows :meth:`list_pots`, so a caller picking "the single
        matching pot" sees the order the per-pot walk produced. Archived pots
        are left out: they are not routing candidates for a repo.
        """
        state = self._load()
        sources = state.get("sources", {})
        rows: list[dict[str, Any]] = []
        for pot_id, pot in state.get("pots", {}).items():
            if pot.get("archived"):
                continue
            for row in sources.get(pot_id, []):
                if row.get("kind") != "repo":
                    continue
                rows.append(
                    {
                        "pot_id": pot_id,
                        "pot_name": pot.get("name", pot_id),
                        "name": row.get("name", row.get("location", "")),
                        "location": row.get("location"),
                    }
                )
        return rows

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            rows = state.get("sources", {}).get(pot_id, [])
            state.setdefault("sources", {})[pot_id] = [
                r for r in rows if r.get("source_id") != source_id
            ]

    # --- repo defaults ------------------------------------------------------
    def repo_default(self, *, repo: str) -> str | None:
        key = _repo_identity_key(repo)
        if not key:
            return None
        value = self._load().get("repo_defaults", {}).get(key)
        return str(value) if value else None

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        key = _repo_identity_key(repo)
        if not key:
            return
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            state.setdefault("repo_defaults", {})[key] = pot_id

    def clear_repo_default(self, *, repo: str) -> bool:
        key = _repo_identity_key(repo)
        if not key:
            return False
        with local_json_transaction(self._path, default_factory=_empty_state) as state:
            defaults = state.setdefault("repo_defaults", {})
            existed = key in defaults
            defaults.pop(key, None)
            return existed

    def list_repo_defaults(self) -> dict[str, str]:
        return {
            str(repo): str(pot_id)
            for repo, pot_id in self._load().get("repo_defaults", {}).items()
        }


def _repo_identity_key(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith((".", "~")) or Path(raw).is_absolute():
        return str(Path(raw).expanduser().resolve(strict=False))
    if raw.endswith(".git"):
        raw = raw[:-4]
    if raw.startswith("git@") and ":" in raw:
        host, path = raw[4:].split(":", 1)
        return f"{host}/{path}".strip("/").lower()
    if "://" in raw:
        from urllib.parse import urlparse

        parsed = urlparse(raw)
        if parsed.netloc and parsed.path:
            return f"{parsed.netloc}/{parsed.path.strip('/')}".lower()
    return raw.strip("/").lower()


__all__ = ["LocalPotStore"]
