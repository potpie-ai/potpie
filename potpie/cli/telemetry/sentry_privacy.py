"""CLI compatibility import for the engine-owned generic Sentry scrubber."""

from potpie_context_engine.bootstrap.sentry_privacy import (
    SentryEvent,
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
)

__all__ = ["SentryEvent", "scrub_sentry_breadcrumb", "scrub_sentry_event"]
