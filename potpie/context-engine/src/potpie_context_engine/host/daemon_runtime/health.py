"""Aggregated health registrar for transports, components, and services."""

from __future__ import annotations

import threading

from potpie_context_engine.domain.ports.daemon.shell import HealthStatus


class HealthRegistrar:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, HealthStatus] = {}

    def set(self, key: str, status: HealthStatus) -> None:
        with self._lock:
            self._states[key] = status

    def get(self, key: str) -> HealthStatus:
        with self._lock:
            return self._states[key]

    def snapshot(self) -> dict[str, HealthStatus]:
        with self._lock:
            return dict(self._states)

    def aggregate(self) -> HealthStatus:
        with self._lock:
            states = list(self._states.values())
        if not states:
            return HealthStatus.STARTING
        if any(state is HealthStatus.STARTING for state in states):
            return HealthStatus.STARTING
        if any(state is HealthStatus.DEGRADED for state in states):
            return HealthStatus.DEGRADED
        if all(state is HealthStatus.READY for state in states):
            return HealthStatus.READY
        return HealthStatus.STOPPED
