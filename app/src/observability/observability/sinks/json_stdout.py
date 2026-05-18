"""Production sink: flat JSONL to stdout (one JSON object per line).

Ported from current logger.py production_log_sink. Edge cases preserved:
 - Flatten extra fields (conversation_id, user_id, ...) to top level so log
   parsers can query them — this is the fix for the audit's 'structure lost'.
 - Serialise exception type/value/traceback; run them through redaction.
 - default=str on json.dumps so non-serialisable extras never crash logging.
 - GAP: stdout JSONL is inert without an aggregator. Where it ships is NOT
   solved by this package (tracked as a separate untracked gap in history).
"""

from __future__ import annotations

import logging

from ..config import ObservabilityConfig


class JsonStdoutSink:
    name = "json_stdout"

    def setup(self, config: ObservabilityConfig) -> None:
        return None

    def build_handler(self, config: ObservabilityConfig) -> logging.Handler | None:
        raise NotImplementedError("Phase 1 scaffold — ported in Phase 2")

    def instrument(self, config: ObservabilityConfig) -> None:
        return None
