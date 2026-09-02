"""The telemetry spool and its detached flusher.

The CLI process writes lines; a child it does not wait for ships them. These
tests pin the file contract (append / take / cap), the single-flusher lock,
the launch shape, and the flusher's shipping — with every network call and
the ``Popen`` replaced by recorders.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from potpie.cli.commands import _common
from potpie.cli.telemetry import flush as flusher
from potpie.cli.telemetry import product_analytics, sentry_runtime, spool
from potpie.cli.telemetry.product_analytics import (
    ProductAnalyticsEvent,
    ProductAnalyticsSettings,
    PostHogSink,
)
from potpie.cli.telemetry.settings import SentrySettings
from potpie_context_engine.bootstrap import sentry_metrics_runtime

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _own_spool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(spool, "_appended", False)
    return spool.spool_path()


def test_append_writes_one_json_line_and_take_drains_it() -> None:
    assert spool.append({"kind": "metric", "name": "a"}) is True
    assert spool.append({"kind": "analytics", "event": {"event": "e"}}) is True

    raw = spool.spool_path().read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    assert all(json.loads(line)["ts"] > 0 for line in raw)
    assert spool.pending_count() == 2

    records = spool.take()

    assert [r["kind"] for r in records] == ["metric", "analytics"]
    assert not spool.spool_path().exists()
    assert spool.pending_count() == 0
    assert spool.take() == []
    # The moved-aside file is gone too: nothing for the next flusher to trip on.
    assert [p.name for p in spool.spool_dir().iterdir()] == []


def test_take_skips_lines_it_cannot_parse() -> None:
    spool.append({"kind": "metric", "name": "a"})
    with spool.spool_path().open("a", encoding="utf-8") as handle:
        handle.write("not json\n\n[1, 2]\n")
    spool.append({"kind": "metric", "name": "b"})

    assert [r["name"] for r in spool.take()] == ["a", "b"]


def test_spool_is_capped_not_grown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline for a week must not turn into a megabyte of stale events."""
    # Two records fit under the cap; the third would push past it.
    monkeypatch.setattr(spool, "SPOOL_MAX_BYTES", 60)

    assert spool.append({"kind": "metric", "name": "first"}) is True
    assert spool.append({"kind": "metric", "name": "second"}) is True
    assert spool.append({"kind": "metric", "name": "third"}) is False
    assert spool.pending_count() == 2


def test_append_survives_an_unwritable_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(blocker))

    assert spool.append({"kind": "metric", "name": "a"}) is False


def test_lock_is_single_flight_and_stale_locks_are_taken_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert spool.acquire_lock() is True
    # Held by this (live) process: a second flusher stands down.
    assert spool.lock_held() is True
    assert spool.acquire_lock() is False
    spool.release_lock()
    assert spool.lock_held() is False

    # A pid that no longer exists is a stale lock, not a live flusher.
    spool.lock_path().write_text("999999999", encoding="ascii")
    assert spool.lock_held() is False
    assert spool.acquire_lock() is True
    spool.release_lock()

    # A live pid but an ancient lock reads as stale too.
    spool.lock_path().write_text(str(os.getpid()), encoding="ascii")
    ancient = time.time() - spool.LOCK_STALE_SECONDS - 10
    os.utime(spool.lock_path(), (ancient, ancient))
    assert spool.lock_held() is False
    assert spool.acquire_lock() is True
    spool.release_lock()


def test_spawn_only_when_there_is_something_to_ship_and_no_flusher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[int] = []
    monkeypatch.setattr(spool, "_launch", lambda: launches.append(1))

    assert spool.spawn_flusher() is False  # nothing spooled
    spool.append({"kind": "metric", "name": "a"})
    assert spool.spawn_flusher() is True
    assert spool.acquire_lock() is True
    try:
        assert spool.spawn_flusher() is False  # a flusher is running
    finally:
        spool.release_lock()
    assert launches == [1]


def test_the_launch_is_detached_and_marked() -> None:
    argv, kwargs = spool.launch_command()

    assert argv == [sys.executable, "-m", "potpie.cli.telemetry.flush"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["close_fds"] is True
    assert kwargs["env"][spool.FLUSHER_ENV] == "1"
    if os.name != "nt":
        assert kwargs["start_new_session"] is True


def test_the_first_append_schedules_one_exit_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    monkeypatch.setattr(spool.atexit, "register", registered.append)

    spool.append({"kind": "metric", "name": "a"})
    spool.append({"kind": "metric", "name": "b"})

    assert registered == [spool._spawn_flusher_at_exit]


def test_a_flusher_never_spawns_a_flusher(monkeypatch: pytest.MonkeyPatch) -> None:
    launches: list[int] = []
    monkeypatch.setattr(spool, "_launch", lambda: launches.append(1))
    monkeypatch.setattr(spool, "exit_spawn_enabled", True)
    monkeypatch.setenv(spool.FLUSHER_ENV, "1")
    spool.append({"kind": "metric", "name": "a"})

    spool._spawn_flusher_at_exit()

    assert launches == []


# --- what the CLI process writes ---------------------------------------------


def test_posthog_sink_spools_the_event_with_its_timestamp() -> None:
    sink = PostHogSink(
        ProductAnalyticsSettings(
            enabled=True, api_key="phc_test", host="https://us.i.posthog.com"
        )
    )

    sink.capture(
        ProductAnalyticsEvent(
            name="cli_usage_command_succeeded",
            distinct_id="install_123",
            properties={"command": "search"},
        )
    )

    (record,) = spool.take()
    assert record["kind"] == "analytics"
    event = record["event"]
    assert event["event"] == "cli_usage_command_succeeded"
    assert event["distinct_id"] == "install_123"
    assert event["properties"] == {"command": "search"}
    assert event["timestamp"].endswith("+00:00")
    # The key is the flusher's to add from settings; it never sits on disk.
    assert "api_key" not in event


def test_a_command_spools_its_metrics_and_never_initialises_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: on the happy path the CLI process does no telemetry
    network work at all — not even ``sentry_sdk.init``."""
    init_calls: list[dict] = []

    class _FakeSentry(type(sys)):
        def __init__(self) -> None:
            super().__init__("sentry_sdk")

        def init(self, **kwargs) -> None:
            init_calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "sentry_sdk", _FakeSentry())
    monkeypatch.setattr(sentry_runtime, "_configured", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_configured", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_enabled", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_sentry_sdk", None)
    settings = SentrySettings(
        enabled=True,
        dsn="https://public@example.invalid/1",
        environment="test",
        release="potpie-cli@test",
        dist=None,
    )

    sentry_runtime.configure_cli_sentry(settings)
    with _common.contract():
        pass

    records = spool.take()
    assert [(r["type"], r["name"]) for r in records] == [
        ("count", "ce.cli.invocations_total"),
        ("distribution", "ce.cli.duration_ms"),
    ]
    assert records[0]["attributes"]["result"] == "ok"
    assert records[1]["unit"] == "millisecond"
    assert init_calls == []
    assert sentry_metrics_runtime.metrics_configured() is False


# --- what the flusher ships ---------------------------------------------------


@pytest.fixture
def _shipping_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    posts: list[tuple[str, dict]] = []
    metrics: list[tuple[str, str, object, str | None, dict | None]] = []
    flushes: list[float] = []
    monkeypatch.setattr(
        product_analytics,
        "_post_product_analytics_batch",
        lambda *, url, payload: posts.append((url, payload)),
    )
    monkeypatch.setattr(product_analytics, "_close_http_client", lambda: None)
    monkeypatch.setattr(
        "potpie.cli.telemetry.settings.load_product_analytics_settings",
        lambda: ProductAnalyticsSettings(
            enabled=True, api_key="phc_test", host="https://us.i.posthog.com"
        ),
    )
    monkeypatch.setattr(
        "potpie.cli.telemetry.settings.load_sentry_settings",
        lambda: SentrySettings(
            enabled=True,
            dsn="https://public@example.invalid/1",
            environment="test",
            release="potpie-cli@test",
            dist=None,
        ),
    )
    monkeypatch.setattr(
        sentry_metrics_runtime, "configure_metrics", lambda s, **kw: None
    )
    monkeypatch.setattr(sentry_metrics_runtime, "metrics_configured", lambda: True)
    for kind in ("count", "distribution", "gauge"):
        monkeypatch.setattr(
            sentry_metrics_runtime,
            kind,
            lambda name, value=1, *, unit=None, attributes=None, _k=kind: (
                metrics.append((_k, name, value, unit, attributes))
            ),
        )
    monkeypatch.setattr(
        sentry_metrics_runtime, "flush", lambda timeout: flushes.append(timeout)
    )
    return {"posts": posts, "metrics": metrics, "flushes": flushes}


def test_flusher_ships_events_as_one_batch_and_metrics_through_the_sdk(
    _shipping_fakes: dict[str, list],
) -> None:
    spool.append({"kind": "analytics", "event": {"event": "one", "distinct_id": "i"}})
    spool.append(
        {
            "kind": "metric",
            "type": "count",
            "name": "ce.cli.invocations_total",
            "value": 1,
            "unit": None,
            "attributes": {"result": "ok"},
        }
    )
    spool.append({"kind": "analytics", "event": {"event": "two", "distinct_id": "i"}})
    spool.append(
        {
            "kind": "metric",
            "type": "distribution",
            "name": "ce.cli.duration_ms",
            "value": 12.5,
            "unit": "millisecond",
            "attributes": {},
        }
    )
    spool.append({"kind": "metric", "type": "bogus", "name": "ignored"})

    assert flusher.main() == 0

    posts = _shipping_fakes["posts"]
    assert len(posts) == 1
    url, payload = posts[0]
    assert url == "https://us.i.posthog.com/batch/"
    assert payload["api_key"] == "phc_test"
    assert [e["event"] for e in payload["batch"]] == ["one", "two"]
    assert _shipping_fakes["metrics"] == [
        ("count", "ce.cli.invocations_total", 1, None, {"result": "ok"}),
        ("distribution", "ce.cli.duration_ms", 12.5, "millisecond", None),
    ]
    assert _shipping_fakes["flushes"] == [5.0]
    assert spool.pending_count() == 0
    assert spool.lock_held() is False


def test_flusher_keeps_going_while_commands_keep_appending(
    _shipping_fakes: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    rounds = {"n": 0}
    real_take = spool.take

    def _take_then_append_once():
        records = real_take()
        if rounds["n"] == 0:
            rounds["n"] += 1
            spool.append({"kind": "analytics", "event": {"event": "late"}})
        return records

    monkeypatch.setattr(spool, "take", _take_then_append_once)
    spool.append({"kind": "analytics", "event": {"event": "early"}})

    flusher.main()

    shipped = [e["event"] for _, p in _shipping_fakes["posts"] for e in p["batch"]]
    assert shipped == ["early", "late"]


def test_flusher_stands_down_when_another_holds_the_lock(
    _shipping_fakes: dict[str, list],
) -> None:
    spool.append({"kind": "analytics", "event": {"event": "one"}})
    assert spool.acquire_lock() is True
    try:
        assert flusher.main() == 0
    finally:
        spool.release_lock()

    assert _shipping_fakes["posts"] == []
    assert spool.pending_count() == 1


def test_flusher_drops_analytics_when_the_sink_is_disabled(
    _shipping_fakes: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "potpie.cli.telemetry.settings.load_product_analytics_settings",
        lambda: ProductAnalyticsSettings(enabled=False, api_key=None, host="h"),
    )
    spool.append({"kind": "analytics", "event": {"event": "one"}})

    flusher.main()

    assert _shipping_fakes["posts"] == []
    assert spool.pending_count() == 0
