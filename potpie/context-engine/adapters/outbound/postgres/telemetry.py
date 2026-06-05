"""SQLAlchemy-backed :class:`TelemetryPort` adapter (Phase 5).

Insert-only writes to ``context_engine_cost_events`` and
``context_engine_drift_snapshots``. The adapter manages its own short-lived
session per call — telemetry must never leak a session reference into a
caller's transaction or fail the request when the DB is briefly unavailable.
Errors are logged at WARNING and swallowed.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

from adapters.outbound.postgres.models import (
    ContextEngineCostEvent,
    ContextEngineDriftSnapshot,
)
from adapters.outbound.postgres.session import database_url, make_session_factory
from domain.ports.telemetry import CostEvent, DriftSnapshot
from domain.error_redaction import safe_error

logger = logging.getLogger(__name__)


class SqlAlchemyTelemetry:
    """Persist cost + drift telemetry to Postgres in an isolated session.

    The session factory is created lazily on first use. If ``DATABASE_URL``
    is not configured the adapter degrades to a silent no-op so callers can
    wire it unconditionally.
    """

    def __init__(self, *, session_factory=None) -> None:
        self._explicit_factory = session_factory
        self._factory = session_factory
        self._initialized = session_factory is not None

    def _ensure_factory(self) -> bool:
        if self._initialized:
            return self._factory is not None
        url = database_url()
        if not url:
            self._initialized = True
            return False
        try:
            self._factory = make_session_factory(url)
        except Exception as exc:
            logger.warning(
                "telemetry: failed to build session factory: %s", safe_error(exc)
            )
            self._factory = None
        self._initialized = True
        return self._factory is not None

    def record_cost(self, event: CostEvent) -> None:
        if not _telemetry_enabled():
            return
        if not self._ensure_factory() or self._factory is None:
            return
        try:
            session = self._factory()
        except Exception as exc:
            logger.warning("telemetry: cost session open failed: %s", safe_error(exc))
            return
        try:
            row = ContextEngineCostEvent(
                id=str(uuid.uuid4()),
                pot_id=event.pot_id,
                kind=event.kind,
                model=event.model,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                total_tokens=event.total_tokens,
                latency_ms=event.latency_ms,
                batch_id=event.batch_id,
                event_id=event.event_id,
                occurred_at=event.occurred_at or datetime.now(timezone.utc),
                metadata_json=dict(event.metadata or {}),
            )
            session.add(row)
            session.commit()
        except Exception as exc:
            logger.warning("telemetry: cost insert failed: %s", safe_error(exc))
            try:
                session.rollback()
            except Exception:
                pass
        finally:
            try:
                session.close()
            except Exception:
                pass

    def record_drift(self, snapshot: DriftSnapshot) -> None:
        if not _telemetry_enabled():
            return
        if not self._ensure_factory() or self._factory is None:
            return
        try:
            session = self._factory()
        except Exception as exc:
            logger.warning("telemetry: drift session open failed: %s", safe_error(exc))
            return
        try:
            row = ContextEngineDriftSnapshot(
                id=str(uuid.uuid4()),
                pot_id=snapshot.pot_id,
                status=snapshot.status,
                source_ref_count=snapshot.source_ref_count,
                stale_ref_count=snapshot.stale_ref_count,
                needs_verification_ref_count=snapshot.needs_verification_ref_count,
                verification_failed_ref_count=snapshot.verification_failed_ref_count,
                source_access_gap_count=snapshot.source_access_gap_count,
                missing_coverage_count=snapshot.missing_coverage_count,
                fallback_count=snapshot.fallback_count,
                open_conflicts_count=snapshot.open_conflicts_count,
                captured_at=snapshot.captured_at or datetime.now(timezone.utc),
                metadata_json=dict(snapshot.metadata or {}),
            )
            session.add(row)
            session.commit()
        except Exception as exc:
            logger.warning("telemetry: drift insert failed: %s", safe_error(exc))
            try:
                session.rollback()
            except Exception:
                pass
        finally:
            try:
                session.close()
            except Exception:
                pass


def _telemetry_enabled() -> bool:
    raw = os.getenv("CONTEXT_ENGINE_TELEMETRY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
