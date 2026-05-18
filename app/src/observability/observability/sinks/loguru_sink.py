"""loguru as an OPTIONAL sink (extra: observability[loguru]).

Note the filename is loguru_sink.py, NOT loguru.py — a module named loguru.py
would shadow the real `loguru` package on import. (Same reason the package is
'observability', not 'logging'.)

Edge cases:
 - Import loguru lazily inside methods; importing this package must not
   require loguru to be installed.
 - loguru<->stdlib bridging: this sink adapts stdlib records INTO loguru's
   sink. Do not also install the InterceptHandler pointing back at loguru, or
   records loop.
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig


class LoguruSink:
    name = "loguru"

    def setup(self, config: ObservabilityConfig) -> None:
        raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
