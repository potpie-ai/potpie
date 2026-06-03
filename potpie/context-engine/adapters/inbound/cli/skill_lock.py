"""Deterministic Potpie skill lockfile handling.

Contract lockfile: `.agents/skills-lock.json`

Shape (v1):
{
  "version": 1,
  "skills": {
    "<skill-id>": {
      "sourceType": "bundled",
      "source": "templates/agent_bundle",
      "templateHash": "sha256:...",
      "installedHash": "sha256:...",
      "installedAt": "...Z",
      "updatedAt": "...Z"
    }
  }
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.inbound.cli.skill_catalog import LOCK_PATH, lock_path

LOCK_VERSION = 1


class SkillLockError(ValueError):
    """Lockfile parse or version error."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def empty_lock() -> dict[str, Any]:
    return {"version": LOCK_VERSION, "skills": {}}


def read_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read lockfile. Returns an empty lock plus diagnostic on parse failures."""

    path = lock_path(root)
    if not path.exists():
        return empty_lock(), None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SkillLockError("Lockfile must contain a JSON object")
        if data.get("version") != LOCK_VERSION:
            raise SkillLockError(
                f"Unsupported lockfile version {data.get('version')!r}"
            )
        skills = data.get("skills")
        if not isinstance(skills, dict):
            raise SkillLockError("Lockfile field 'skills' must be an object")
        return {"version": LOCK_VERSION, "skills": dict(skills)}, None
    except Exception as exc:
        return empty_lock(), {
            "code": "INVALID_LOCKFILE",
            "path": LOCK_PATH.as_posix(),
            "message": str(exc),
        }


def write_lock(root: Path, data: dict[str, Any]) -> None:
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "version": LOCK_VERSION,
        "skills": {
            skill_id: data.get("skills", {})[skill_id]
            for skill_id in sorted(data.get("skills", {}))
        },
    }
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def upsert_lock_entry(
    lock: dict[str, Any],
    *,
    skill_id: str,
    source: str,
    skill_path: str,
    template_hash: str,
    installed_hash: str,
) -> None:
    now = utc_now()
    skills = lock.setdefault("skills", {})
    previous = skills.get(skill_id)
    installed_at = (
        previous.get("installedAt")
        if isinstance(previous, dict) and previous.get("installedAt")
        else now
    )
    skills[skill_id] = {
        "sourceType": "bundled",
        "source": source,
        "skillPath": skill_path,
        "templateHash": template_hash,
        "installedHash": installed_hash,
        "installedAt": installed_at,
        "updatedAt": now,
    }


def remove_lock_entry(lock: dict[str, Any], skill_id: str) -> None:
    skills = lock.setdefault("skills", {})
    if isinstance(skills, dict):
        skills.pop(skill_id, None)
