"""Lightweight metrics sink for sandbox tool calls.

Kept deliberately small: a single ``record(name, labels, value)`` interface
that the sandbox tool builders can call inline without dragging the full
:class:`TelemetryPort` (cost / drift) onto the call site. Defaults to a
logger sink (debug-level so it doesn't spam prod), swappable to PostHog /
Prometheus via :func:`set_sink`.

Event names follow the plan:

- ``pot_sandbox.attach`` — first-time workspace acquire per (pot, repo)
- ``pot_sandbox.checkout`` — successful ``sandbox_checkout``
- ``pot_sandbox.cold_start_ms`` — ms spent on first ``acquire`` per batch
- ``pot_sandbox.tool_call`` — every sandbox tool call; ``labels={"name":..., "ok":...}``
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


SandboxMetricsSink = Callable[[str, Mapping[str, Any], float], None]


def _default_sink(name: str, labels: Mapping[str, Any], value: float) -> None:
    logger.debug("sandbox.metric %s value=%s labels=%s", name, value, dict(labels))


_SINK: SandboxMetricsSink = _default_sink


def set_sink(sink: SandboxMetricsSink | None) -> None:
    """Swap the metrics sink process-wide. ``None`` restores the default."""
    global _SINK
    _SINK = sink or _default_sink


def record(name: str, labels: Mapping[str, Any] | None = None, value: float = 1.0) -> None:
    """Emit a metric. Never raises — telemetry must not break the tool path."""
    try:
        _SINK(name, dict(labels or {}), float(value))
    except Exception:  # noqa: BLE001
        logger.debug("sandbox metric %s emit failed", name, exc_info=True)
