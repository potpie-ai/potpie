"""Development sink: human-readable colorized line. Ported from logger.py dev
sink. Edge case: appends extra fields to the message tail for readability;
must still pass through redaction before write.
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig


class ConsoleSink:
    name = "console"

    def setup(self, config: ObservabilityConfig) -> None:
        return None

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
