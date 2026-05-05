"""No-op eviction policy.

Default for tests and standalone use where disk pressure is not a
concern. Production deployments should compose a volume- or age-based
policy instead.
"""

from __future__ import annotations

from sandbox.domain.ports.eviction import EvictionResult


class NoOpEvictionPolicy:
    """Eviction policy that never evicts."""

    async def evict_if_needed(
        self, *, user_id: str | None = None
    ) -> EvictionResult:
        _ = user_id
        return EvictionResult()
