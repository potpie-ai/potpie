"""logfire log sink (extra: observability[logfire]) — NEW.

logfire serves double duty: tracing (observability/tracing.py) AND structured
log export (this sink). Both MUST share a single logfire.configure() call —
setup() here delegates to tracing.configure_tracing() if not already done,
never configures logfire twice (EC2).

Edge cases:
 - Import logfire lazily.
 - When this sink is active, log records should carry the active trace/span
   id so logs and traces correlate in one backend (the unification goal).
 - No token -> local-only, single visible notice (no silent no-op).
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig


class LogfireSink:
    name = "logfire"

    def setup(self, config: ObservabilityConfig) -> None:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")

    def instrument(self, config: ObservabilityConfig) -> None:
        raise NotImplementedError("Phase 1 scaffold — implemented in Phase 3")
