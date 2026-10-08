"""What one install target wrote: one manifest file per target, under the Potpie home.

``skill_manifest_<agent>_<scope>.json`` — with a per-repository suffix at
project scope — maps each skill id to what install recorded for it: the
``version`` written, the ``sha256`` of the ``SKILL.md`` written, and whether it
is ``disabled`` (removed by id, so bundle installs skip it).

It is a *cache* of what install last wrote; the files on disk are the truth
about what is installed. So an unreadable manifest, or an entry of the wrong
shape, reads as "nothing recorded": a present skill then reports
``installed_version="unknown"``, lands in ``skills status`` as outdated, and
is repaired by the reinstall that report already names.

Earlier releases kept the same facts in three files per target —
``skills_<stem>.json`` (versions), ``skill_hashes_<stem>.json`` and
``skill_disabled_<stem>.json``. The first read of a target that has those but
no manifest folds whatever is readable into a new manifest, writes it
atomically, and only then deletes them.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

MANIFEST_PREFIX = "skill_manifest_"
#: ``(prefix, field)`` for each per-target file earlier releases wrote.
_LEGACY_FILES = (
    ("skills_", "version"),
    ("skill_hashes_", "sha256"),
    ("skill_disabled_", "disabled"),
)

Record = dict[str, Any]


def _read_json(path: Path) -> Mapping[str, Any]:
    """A JSON object from ``path``, or an empty one for anything unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, Mapping) else {}


def _clean_record(raw: object) -> Record:
    """Keep the fields a record may hold, in the types it may hold them."""
    if not isinstance(raw, Mapping):
        return {}
    record: Record = {}
    for key in ("version", "sha256"):
        if raw.get(key) is not None:
            record[key] = str(raw[key])
    if raw.get("disabled") is True:
        record["disabled"] = True
    return record


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SkillManifest:
    """The one manifest for the target whose files are named by ``stem``."""

    def __init__(self, home: Path, stem: str) -> None:
        self.home = home
        self.stem = stem

    @property
    def path(self) -> Path:
        return self.home / f"{MANIFEST_PREFIX}{self.stem}.json"

    def legacy_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.home / f"{prefix}{self.stem}.json" for prefix, _ in _LEGACY_FILES
        )

    def read(self) -> dict[str, Record]:
        """``skill id -> record``, migrating the older per-target files first."""
        if not self.path.exists():
            migrated = self._migrate()
            if migrated is not None:
                return migrated
        skills = _read_json(self.path).get("skills")
        if not isinstance(skills, Mapping):
            return {}
        records = {str(sid): _clean_record(raw) for sid, raw in skills.items()}
        return {sid: record for sid, record in records.items() if record}

    def update(self, change: Callable[[dict[str, Record]], None]) -> None:
        """Apply ``change`` to the current records and write them back atomically."""
        records = self.read()
        change(records)
        self._write(records)

    def _write(self, records: Mapping[str, Record]) -> None:
        skills = {sid: record for sid, record in sorted(records.items()) if record}
        _atomic_write_json(self.path, {"format": 1, "skills": skills})

    def _migrate(self) -> dict[str, Record] | None:
        """Fold the older files into a manifest; ``None`` if there are none.

        A corrupt or partial old file contributes what it can, which may be
        nothing. The old files are deleted only once the manifest is on disk,
        so a failed write leaves them for the next read to try again.
        """
        legacy = [
            (path, field)
            for path, (_, field) in zip(self.legacy_paths(), _LEGACY_FILES)
            if path.exists()
        ]
        if not legacy:
            return None
        records: dict[str, Record] = {}
        for path, field in legacy:
            for sid, value in _read_json(path).items():
                record = records.setdefault(str(sid), {})
                record[field] = True if field == "disabled" else str(value)
        try:
            self._write(records)
        except OSError:
            return records
        for path, _ in legacy:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # Superseded either way: the manifest wins from now on.
        return records


__all__ = ["MANIFEST_PREFIX", "SkillManifest"]
