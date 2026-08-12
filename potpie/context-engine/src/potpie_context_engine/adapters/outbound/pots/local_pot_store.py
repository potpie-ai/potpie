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
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.domain.ports.services.pot_management import (
    INGESTION_NOT_STARTED,
    SOURCE_REGISTERED,
)
from potpie_context_engine.domain.repo_identity import repo_identity_key


def default_home() -> Path:
    raw = os.getenv("CONTEXT_ENGINE_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".potpie"


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
            return {"pots": {}, "active": None, "sources": {}, "repo_defaults": {}}

    def _save(self, state: dict[str, Any]) -> None:
        """Replace ``pots.json`` atomically, through a temp file of its own.

        The scratch file used to be the fixed ``pots.tmp``, which two writers in
        flight both opened: the daemon serves ``POST /ui/api/pots/use`` off a
        threadpool and the CLI is a second process entirely, so the first
        ``replace`` moved the shared temp out from under the second and the
        loser raised ``FileNotFoundError`` — a bare 500 in the explorer for an
        operation that had simply been overtaken.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=self.home, prefix=f".{self._path.name}.", suffix=".tmp"
        )
        tmp = Path(temp_name)
        try:
            # fdopen takes ownership of the descriptor immediately, so nothing
            # between here and the close can leak it.
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            tmp.replace(self._path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

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
        state = self._load()
        # Reuse an existing pot by name (idempotent setup) — but never an
        # archived one. Archiving tore its graph and resource tree down, so
        # "reuse" would hand back an empty pot under a name the caller expected
        # to be new, and quietly un-hide it by making it active.
        for pid, row in state.get("pots", {}).items():
            if row.get("name") == name and not row.get("archived"):
                if use:
                    state["active"] = pid
                    self._save(state)
                return {**row, "active": state.get("active") == pid, "created": False}
        pot_id = f"pot_{uuid.uuid4().hex[:12]}"
        row = {"pot_id": pot_id, "name": name, "archived": False}
        state.setdefault("pots", {})[pot_id] = row
        if use or state.get("active") is None:
            state["active"] = pot_id
        self._save(state)
        return {**row, "active": state.get("active") == pot_id, "created": True}

    def _resolve_ref(
        self, state: dict[str, Any], ref: str, *, include_archived: bool = False
    ) -> str | None:
        """The pot id a ref names.

        Archived pots are excluded by default: they are not selectable, not
        writable, and not routable, so leaving them in the id/name space means a
        live pot can be shadowed by a dead one. Callers that need to *see* an
        archived pot — the listing, and the lookup that decides which refusal to
        raise — opt in.
        """
        pots = state.get("pots", {})

        def _eligible(pid: str) -> bool:
            return include_archived or not pots[pid].get("archived")

        if ref in pots and _eligible(ref):
            return ref
        for pid, row in pots.items():
            if row.get("name") == ref and _eligible(pid):
                return pid
        return None

    def find(self, *, ref: str, include_archived: bool = True) -> dict[str, Any] | None:
        """The row a ref names, without selecting or mutating anything."""
        state = self._load()
        pid = self._resolve_ref(state, ref, include_archived=include_archived)
        if pid is None:
            return None
        return {**state["pots"][pid], "active": state.get("active") == pid}

    def names_in_use(self) -> dict[str, str]:
        """``name -> pot_id`` for every live pot, for uniqueness checks."""
        state = self._load()
        return {
            str(row.get("name")): pid
            for pid, row in state.get("pots", {}).items()
            if not row.get("archived") and row.get("name")
        }

    def pot_ids(self) -> frozenset[str]:
        return frozenset(self._load().get("pots", {}))

    def use(self, *, ref: str) -> dict[str, Any] | None:
        state = self._load()
        pid = self._resolve_ref(state, ref)
        if pid is None:
            return None
        state["active"] = pid
        self._save(state)
        return {**state["pots"][pid], "active": True}

    def rename(self, *, ref: str, new_name: str) -> dict[str, Any] | None:
        state = self._load()
        pid = self._resolve_ref(state, ref)
        if pid is None:
            return None
        state["pots"][pid]["name"] = new_name
        self._save(state)
        return {**state["pots"][pid], "active": state.get("active") == pid}

    def archive(self, *, ref: str) -> dict[str, Any] | None:
        state = self._load()
        pid = self._resolve_ref(state, ref)
        if pid is None:
            return None
        state["pots"][pid]["archived"] = True
        if state.get("active") == pid:
            state["active"] = None
        self._save(state)
        return {**state["pots"][pid], "active": False}

    # --- sources ------------------------------------------------------------
    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> dict[str, Any]:
        """Persist one source row, including the two fields that report on it.

        ``status`` and ``ingestion_status`` are written here rather than
        synthesized when the row is read. They were literals two layers up —
        ``source status`` printed ``ok`` / ``not started`` for every row no
        matter what the row said — so the two fields a user consults to find out
        whether a source is healthy or ingested could not carry an answer at
        all. Written, they can: whatever later marks a source stale, broken, or
        ingested updates the row, and the reader reports it.
        """
        state = self._load()
        row = {
            "source_id": f"src_{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "name": name or location,
            "location": location,
            "status": SOURCE_REGISTERED,
            "ingestion_status": INGESTION_NOT_STARTED,
        }
        state.setdefault("sources", {}).setdefault(pot_id, []).append(row)
        self._save(state)
        return row

    def list_sources(self, *, pot_id: str) -> list[dict[str, Any]]:
        return self._load().get("sources", {}).get(pot_id, [])

    def list_repo_sources(self) -> list[dict[str, Any]]:
        """Repo sources of every pot, joined to their pot, from one load.

        Pot order follows :meth:`list_pots` so callers that pick "the single
        matching pot" see the same ordering they got from the per-pot walk.
        """
        state = self._load()
        sources = state.get("sources", {})
        rows: list[dict[str, Any]] = []
        for pot_id, pot in state.get("pots", {}).items():
            # An archived pot is not a routing candidate. Left in, a repo whose
            # pot had been archived still resolved to it, and every repo-scoped
            # read and write went into a pot whose graph had been torn down.
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

    def remove_source(self, *, pot_id: str, source_id: str) -> bool:
        """Drop one source row; report whether one was actually there.

        A filter that quietly rewrites the list to itself cannot be told apart
        from a removal, so an id belonging to another pot — or to no pot —
        looked like a success all the way out to the CLI.
        """
        state = self._load()
        rows = state.get("sources", {}).get(pot_id, [])
        kept = [r for r in rows if r.get("source_id") != source_id]
        if len(kept) == len(rows):
            return False
        state.setdefault("sources", {})[pot_id] = kept
        self._save(state)
        return True

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
        state = self._load()
        state.setdefault("repo_defaults", {})[key] = pot_id
        self._save(state)

    def clear_repo_default(self, *, repo: str) -> bool:
        key = _repo_identity_key(repo)
        if not key:
            return False
        state = self._load()
        defaults = state.setdefault("repo_defaults", {})
        existed = key in defaults
        defaults.pop(key, None)
        if existed:
            self._save(state)
        return existed

    def list_repo_defaults(self) -> dict[str, str]:
        return {
            str(repo): str(pot_id)
            for repo, pot_id in self._load().get("repo_defaults", {}).items()
        }


# Every ``repo_defaults`` key on every user's disk came out of this function, so
# it moved to the domain rather than being reimplemented next to the callers that
# also need it (the setup seam had its own drifted copy). Kept as a module-level
# alias so the on-disk keys cannot change by accident.
_repo_identity_key = repo_identity_key


__all__ = ["LocalPotStore", "default_home"]
