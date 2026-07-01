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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        tmp.replace(self._path)

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
        # Reuse an existing pot by name (idempotent setup).
        for pid, row in state.get("pots", {}).items():
            if row.get("name") == name:
                if use:
                    state["active"] = pid
                    self._save(state)
                return {**row, "active": state.get("active") == pid}
        pot_id = f"pot_{uuid.uuid4().hex[:12]}"
        row = {"pot_id": pot_id, "name": name, "archived": False}
        state.setdefault("pots", {})[pot_id] = row
        if use or state.get("active") is None:
            state["active"] = pot_id
        self._save(state)
        return {**row, "active": state.get("active") == pot_id}

    def _resolve_ref(self, state: dict[str, Any], ref: str) -> str | None:
        if ref in state.get("pots", {}):
            return ref
        for pid, row in state.get("pots", {}).items():
            if row.get("name") == ref:
                return pid
        return None

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
        state = self._load()
        row = {
            "source_id": f"src_{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "name": name or location,
            "location": location,
        }
        state.setdefault("sources", {}).setdefault(pot_id, []).append(row)
        self._save(state)
        return row

    def list_sources(self, *, pot_id: str) -> list[dict[str, Any]]:
        return self._load().get("sources", {}).get(pot_id, [])

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        state = self._load()
        rows = state.get("sources", {}).get(pot_id, [])
        state.setdefault("sources", {})[pot_id] = [
            r for r in rows if r.get("source_id") != source_id
        ]
        self._save(state)

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


__all__ = ["LocalPotStore", "default_home"]
