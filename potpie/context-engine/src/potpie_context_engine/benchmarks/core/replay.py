"""Load + timestamp-rewrite + replay fixture envelopes.

The replay layer is the bridge between scenario YAML (which references
fixtures by relative path with a relative timestamp like ``-60d``) and
the engine's HTTP submission API.

Each ingest / seed / distractor step is converted to one or more
``ReplayEvent``s with:

- the loaded envelope (from ``fixtures/raw_events/<connector>/<file>``
  or ``universe/<name>/...`` for seed events)
- a concrete ``occurred_at`` timestamp anchored to the scenario's
  reference time (default = now)

Distractor steps may expand into ``count`` events, either by enumerating
files matching a glob pattern or by templating a single fixture into N
near-duplicates with rotated identifiers (so reconciliation actually
sees N distinct events).
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from potpie_context_engine.benchmarks.core.scenario import DistractorStep, IngestStep, SeedStep

_OFFSET_RE = re.compile(r"^([+-]?\d+)([smhd])$")
_OFFSET_RANGE_RE = re.compile(r"^([+-]?\d+)([smhd])\.\.([+-]?\d+)([smhd])$")


class FixtureNotFound(FileNotFoundError):
    pass


class FixtureValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayEvent:
    fixture_path: str  # Relative reference, e.g. "github/pr_merged__1042.json"
    connector: str  # github | linear | slack | notion | repo_docs | alerting | deploy
    event_type: str
    action: str
    source_id: str
    repo_name: str | None
    occurred_at: datetime
    payload: dict[str, Any]
    # Author classification — "signal" / "distractor" / "seed". Used by
    # precision/coverage evaluators to know what was expected vs. noise.
    role: str = "signal"


def _unit_delta(amount: int, unit: str) -> timedelta:
    return {
        "s": timedelta(seconds=amount),
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
        "d": timedelta(days=amount),
    }[unit]


def _resolve_offset(at: str, anchor: datetime) -> datetime:
    match = _OFFSET_RE.match(at)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        return anchor + _unit_delta(amount, unit)
    # ISO 8601 absolute.
    parsed = datetime.fromisoformat(at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_offset_range(at: str, anchor: datetime, n: int) -> list[datetime]:
    """Spread N timestamps uniformly across a relative offset range."""
    match = _OFFSET_RANGE_RE.match(at)
    if not match:
        # Single offset → broadcast.
        return [_resolve_offset(at, anchor)] * n
    start_amount = int(match.group(1))
    start_unit = match.group(2)
    end_amount = int(match.group(3))
    end_unit = match.group(4)
    start = anchor + _unit_delta(start_amount, start_unit)
    end = anchor + _unit_delta(end_amount, end_unit)
    if n <= 1 or start == end:
        return [start]
    span = (end - start) / max(1, n - 1)
    return [start + span * i for i in range(n)]


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
            raise FixtureValidationError(
                f"{source}: envelope missing required key '{key}'"
            )
    if not isinstance(envelope["payload"], dict):
        raise FixtureValidationError(f"{source}: 'payload' must be a JSON object")


def _envelope_to_replay(
    envelope: dict[str, Any],
    *,
    fixture_path: str,
    occurred_at: datetime,
    role: str,
    source_id_override: str | None = None,
) -> ReplayEvent:
    return ReplayEvent(
        fixture_path=fixture_path,
        connector=str(envelope["connector"]),
        event_type=str(envelope["event_type"]),
        action=str(envelope["action"]),
        source_id=source_id_override or str(envelope["source_id"]),
        repo_name=envelope.get("repo_name"),
        occurred_at=occurred_at,
        payload=dict(envelope["payload"]),
        role=role,
    )


def to_replay_event(
    step: IngestStep, fixtures_root: Path, *, anchor: datetime
) -> ReplayEvent:
    envelope = load_envelope(fixtures_root, step.event)
    validate_envelope(envelope, step.event)
    role = "signal"
    return _envelope_to_replay(
        envelope,
        fixture_path=step.event,
        occurred_at=_resolve_offset(step.at, anchor),
        role=role,
    )


def to_seed_event(
    step: SeedStep, fixtures_root: Path, *, anchor: datetime
) -> ReplayEvent:
    """Resolve a seed step to a single replay event.

    Seed envelopes live under ``fixtures/raw_events/universe/<name>/...``
    and follow the same envelope schema as signal events.
    """
    envelope = load_envelope(fixtures_root, step.event)
    validate_envelope(envelope, step.event)
    return _envelope_to_replay(
        envelope,
        fixture_path=step.event,
        occurred_at=_resolve_offset(step.at, anchor),
        role="seed",
    )


def _expand_distractor_files(fixtures_root: Path, pattern: str) -> list[str]:
    """Resolve a fixture-relative glob to a sorted list of fixture paths.

    Patterns that contain no wildcard characters return ``[pattern]``
    (the single literal path).
    """
    if not any(ch in pattern for ch in "*?["):
        return [pattern]
    base = fixtures_root / "raw_events"
    matches = sorted(p.relative_to(base) for p in base.glob(pattern))
    return [str(m).replace("\\", "/") for m in matches]


def expand_distractor(
    step: DistractorStep, fixtures_root: Path, *, anchor: datetime
) -> list[ReplayEvent]:
    """Expand a single distractor declaration into N replay events.

    Three modes:

    - ``count == 1`` and no wildcard → load that single envelope.
    - Wildcard pattern → enumerate matching files; ``count`` caps the list.
    - Single fixture + ``count > 1`` → load once and clone N times with
      rotated ``source_id`` + ``payload.data.id`` so the engine treats
      them as distinct events.
    """
    matched = _expand_distractor_files(fixtures_root, step.event)
    if not matched:
        raise FixtureNotFound(f"distractor pattern matched no fixtures: {step.event}")

    if len(matched) >= step.count and len(matched) > 1:
        # Enumeration mode — N files, take first ``count``.
        chosen = matched[: step.count]
        times = _resolve_offset_range(step.at, anchor, len(chosen))
        out: list[ReplayEvent] = []
        for fp, t in zip(chosen, times):
            envelope = load_envelope(fixtures_root, fp)
            validate_envelope(envelope, fp)
            out.append(
                _envelope_to_replay(
                    envelope, fixture_path=fp, occurred_at=t, role="distractor"
                )
            )
        return out

    # Clone mode — one fixture, replicate ``count`` times with rotated ids.
    fp = matched[0]
    envelope = load_envelope(fixtures_root, fp)
    validate_envelope(envelope, fp)
    times = _resolve_offset_range(step.at, anchor, step.count)
    out_clones: list[ReplayEvent] = []
    for i, t in enumerate(times):
        cloned = copy.deepcopy(envelope)
        suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"{fp}#{i}").hex[:12]
        cloned["source_id"] = f"{envelope['source_id']}::dup-{suffix}"
        data = cloned.get("payload", {}).get("data")
        if isinstance(data, dict) and "id" in data:
            data["id"] = f"{data['id']}-{suffix}"
        out_clones.append(
            _envelope_to_replay(
                cloned,
                fixture_path=f"{fp}#dup-{suffix}",
                occurred_at=t,
                role="distractor",
                source_id_override=cloned["source_id"],
            )
        )
    return out_clones


def to_replay_events(
    steps: tuple[IngestStep, ...],
    fixtures_root: Path,
    *,
    anchor: datetime | None = None,
) -> list[ReplayEvent]:
    if anchor is None:
        anchor = datetime.now(timezone.utc)
    return [to_replay_event(s, fixtures_root, anchor=anchor) for s in steps]


def assemble_timeline(
    *,
    seed_steps: tuple[SeedStep, ...],
    ingest_steps: tuple[IngestStep, ...],
    distractor_steps: tuple[DistractorStep, ...],
    fixtures_root: Path,
    anchor: datetime | None = None,
) -> list[ReplayEvent]:
    """Build the ordered ingestion timeline for a scenario.

    Order is by resolved ``occurred_at``, ascending. Within the same
    timestamp, signal events come before distractors (lets the engine
    see canonical state first, then noise, which is the harder case to
    pass).
    """
    if anchor is None:
        anchor = datetime.now(timezone.utc)
    events: list[ReplayEvent] = []
    for s in seed_steps:
        events.append(to_seed_event(s, fixtures_root, anchor=anchor))
    for s in ingest_steps:
        events.append(to_replay_event(s, fixtures_root, anchor=anchor))
    for d in distractor_steps:
        events.extend(expand_distractor(d, fixtures_root, anchor=anchor))
    # Stable: by time then role-priority (signal < distractor < seed).
    role_rank = {"seed": 0, "signal": 1, "distractor": 2}
    events.sort(key=lambda e: (e.occurred_at, role_rank.get(e.role, 3)))
    return events


def build_source_id_index(fixtures_root: Path) -> dict[str, str]:
    """Build a map of ``fixture_path -> source_id`` for the citation matcher.

    The retrieval evaluator uses this so ``must_cite_event_id`` resolves
    to the canonical engine identifier rather than the fixture filename.
    """
    out: dict[str, str] = {}
    raw_root = fixtures_root / "raw_events"
    if not raw_root.exists():
        return out
    for path in raw_root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                envelope = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(envelope, dict):
            continue
        sid = envelope.get("source_id")
        if not isinstance(sid, str):
            continue
        rel = path.relative_to(raw_root).as_posix()
        out[rel] = sid
    return out


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
