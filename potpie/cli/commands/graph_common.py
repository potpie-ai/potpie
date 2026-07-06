"""Shared parsing helpers for graph CLI commands."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from potpie.cli.repo_location import current_git_remote, normalize_repo_ref


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _parse_scope(scope: str | None) -> dict[str, str]:
    if not scope:
        return {}
    out: dict[str, str] = {}
    for pair in scope.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"invalid --scope entry {pair!r}; expected key:value pairs"
            )
        key, value = pair.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"invalid --scope entry {pair!r}; scope keys must not be empty"
            )
        value = value.strip()
        if not value:
            raise ValueError(
                f"invalid --scope entry {pair!r}; scope values must not be empty"
            )
        out[key] = value
    return out


def _parse_created_by(value: str | None) -> dict[str, Any]:
    clean = value.strip() if isinstance(value, str) else ""
    if not clean:
        return {"surface": "cli"}
    if clean.startswith("{"):
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --created-by JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--created-by JSON must be an object")
        parsed.setdefault("surface", "cli")
        return parsed
    return {"surface": "cli", "actor": clean}


def _parse_predicates(predicate: str | None) -> tuple[str, ...]:
    if not predicate:
        return ()
    out: list[str] = []
    for raw in predicate.split(","):
        value = raw.strip().upper()
        if value:
            out.append(value)
    return tuple(dict.fromkeys(out))


def _parse_instant(value: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise ValueError("timestamp must be non-empty")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp {value!r}; expected ISO 8601") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_duration(value: str) -> timedelta:
    m = re.fullmatch(r"\s*(\d+)\s*([mhdw])\s*", value.strip().lower())
    if not m:
        raise ValueError("--time-window must look like 30m, 24h, 7d, or 2w")
    amount = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    return timedelta(weeks=amount)


def _parse_ttl_seconds(value: str) -> int:
    try:
        ttl = _parse_duration(value)
    except ValueError as exc:
        raise ValueError("--ttl must look like 30m, 1h, 7d, or 2w") from exc
    seconds = int(ttl.total_seconds())
    if seconds <= 0:
        raise ValueError("--ttl must be positive")
    return seconds


def _parse_sort_dt(value: Any) -> datetime:
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            raw = raw + "T00:00:00+00:00"
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _resolve_repo_scope(repo: str) -> str:
    value = repo.strip()
    if not value:
        raise ValueError("--repo must be non-empty")
    if value == "current":
        remote = _current_repo_remote_for_scope()
        if remote:
            return remote
        raise ValueError("--repo current requires a git remote.origin.url")
    return _normalize_repo_for_scope(value)


def _current_repo_remote_for_scope() -> str | None:
    return current_git_remote(Path.cwd())


def _normalize_repo_for_scope(value: str) -> str:
    return normalize_repo_ref(value) or ""


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str) and v]
    return []
