"""Load + timestamp-rewrite + replay fixture envelopes.

The replay layer is the bridge between scenario YAML (which references
fixtures by relative path with a relative timestamp like ``-60d``) and
the engine's HTTP submission API.

Each ``IngestStep`` is converted to a ``ReplayEvent`` with:
- the loaded envelope (from ``fixtures/raw_events/<connector>/<file>``)
- a concrete ``occurred_at`` timestamp (anchored to the scenario's
  reference time, default = now)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.core.scenario import IngestStep

_OFFSET_RE = re.compile(r"^([+-]?\d+)([smhd])$")


class FixtureNotFound(FileNotFoundError):
    pass


class FixtureValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayEvent:
    fixture_path: str  # Relative reference, e.g. "github/pr_merged__1042.json"
    connector: str  # github | linear
    event_type: str  # pull_request | issue | ...
    action: str  # opened | merged | created | ...
    source_id: str
    repo_name: str | None
    occurred_at: datetime
    payload: dict[str, Any]


def _resolve_offset(at: str, anchor: datetime) -> datetime:
    match = _OFFSET_RE.match(at)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
        }[unit]
        return anchor + delta
    # ISO 8601 absolute.
    parsed = datetime.fromisoformat(at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_envelope(fixtures_root: Path, fixture_path: str) -> dict[str, Any]:
    """Load a fixture envelope by its scenario-relative path."""
    full = fixtures_root / "raw_events" / fixture_path
    if not full.exists():
        raise FixtureNotFound(f"fixture not found: {full}")
    with full.open("r", encoding="utf-8") as f:
        envelope = json.load(f)
    if not isinstance(envelope, dict):
        raise FixtureValidationError(f"{full}: envelope must be a JSON object")
    return envelope


REQUIRED_ENVELOPE_KEYS = ("connector", "event_type", "action", "source_id", "payload")


def validate_envelope(envelope: dict[str, Any], source: Path | str) -> None:
    """Validate a fixture envelope. Raises on schema violation."""
    for key in REQUIRED_ENVELOPE_KEYS:
        if key not in envelope:
            raise FixtureValidationError(f"{source}: envelope missing required key '{key}'")
    if not isinstance(envelope["payload"], dict):
        raise FixtureValidationError(f"{source}: 'payload' must be a JSON object")


def to_replay_event(
    step: IngestStep, fixtures_root: Path, *, anchor: datetime
) -> ReplayEvent:
    envelope = load_envelope(fixtures_root, step.event)
    validate_envelope(envelope, step.event)
    return ReplayEvent(
        fixture_path=step.event,
        connector=str(envelope["connector"]),
        event_type=str(envelope["event_type"]),
        action=str(envelope["action"]),
        source_id=str(envelope["source_id"]),
        repo_name=envelope.get("repo_name"),
        occurred_at=_resolve_offset(step.at, anchor),
        payload=dict(envelope["payload"]),
    )


def to_replay_events(
    steps: tuple[IngestStep, ...], fixtures_root: Path, *, anchor: datetime | None = None
) -> list[ReplayEvent]:
    if anchor is None:
        anchor = datetime.now(timezone.utc)
    return [to_replay_event(s, fixtures_root, anchor=anchor) for s in steps]


def validate_all_fixtures(fixtures_root: Path) -> list[str]:
    """Walk all fixture envelopes and return a list of validation errors."""
    errors: list[str] = []
    raw_root = fixtures_root / "raw_events"
    if not raw_root.exists():
        return errors
    for path in sorted(raw_root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                envelope = json.load(f)
            if not isinstance(envelope, dict):
                errors.append(f"{path}: envelope must be a JSON object")
                continue
            validate_envelope(envelope, path)
        except (json.JSONDecodeError, FixtureValidationError) as exc:
            errors.append(f"{path}: {exc}")
    return errors
