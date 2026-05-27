"""CGT-7: stale-batch reaper lease must exceed Celery hard time limit.

The reaper presumes a batch is dead only when ``claimed_at`` is older than
the lease. If the lease were shorter than ``CELERY_TASK_TIME_LIMIT``, a slow
but live run could be failed while still executing — double work and corrupt
terminal state. Default lease is ``CELERY_TASK_TIME_LIMIT + 900`` (15 min
headroom).
"""

from __future__ import annotations

import pytest

from app.modules.context_graph.tasks import _stale_batch_lease_seconds

pytestmark = pytest.mark.unit


class TestStaleBatchReaperLease:
    def test_default_lease_is_task_time_limit_plus_900(self, monkeypatch) -> None:
        monkeypatch.delenv("CELERY_STALE_BATCH_LEASE_SECS", raising=False)
        monkeypatch.delenv("CELERY_TASK_TIME_LIMIT", raising=False)
        assert _stale_batch_lease_seconds() == 5400 + 900

    def test_lease_tracks_custom_task_time_limit(self, monkeypatch) -> None:
        monkeypatch.delenv("CELERY_STALE_BATCH_LEASE_SECS", raising=False)
        monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", "3600")
        assert _stale_batch_lease_seconds() == 3600 + 900

    def test_explicit_lease_override(self, monkeypatch) -> None:
        monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", "5400")
        monkeypatch.setenv("CELERY_STALE_BATCH_LEASE_SECS", "7200")
        assert _stale_batch_lease_seconds() == 7200.0

    def test_default_lease_strictly_exceeds_task_time_limit(self, monkeypatch) -> None:
        """Headroom invariant: reaper must not run ahead of Celery hard-kill."""
        monkeypatch.delenv("CELERY_STALE_BATCH_LEASE_SECS", raising=False)
        for limit in ("1800", "5400", "7200"):
            monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", limit)
            lease = _stale_batch_lease_seconds()
            assert lease > int(limit)
