from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
import sys
from types import ModuleType
from typing import Optional, Union

import pytest

from potpie_context_engine.adapters.inbound.cli.telemetry.settings import SentrySettings

sentry_metrics_runtime = import_module("potpie_context_engine.bootstrap.sentry_metrics_runtime")
runtime_importlib = getattr(sentry_metrics_runtime, "importlib")
configure_metrics = getattr(sentry_metrics_runtime, "configure_metrics")
count = getattr(sentry_metrics_runtime, "count")
distribution = getattr(sentry_metrics_runtime, "distribution")
gauge = getattr(sentry_metrics_runtime, "gauge")
flush = getattr(sentry_metrics_runtime, "flush")


@dataclass(frozen=True)
class _MetricCall:
    name: str
    value: Union[int, float]
    unit: Optional[str]
    attributes: Mapping[str, Union[str, int, float, bool]]


class _FakeMetrics:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.count_calls: list[_MetricCall] = []
        self.distribution_calls: list[_MetricCall] = []
        self.gauge_calls: list[_MetricCall] = []

    def count(
        self,
        name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        attributes: Optional[dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        if self.fail:
            raise RuntimeError("sdk counter failed")
        self.count_calls.append(_call(name, value, unit, attributes))

    def distribution(
        self,
        name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        attributes: Optional[dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        if self.fail:
            raise RuntimeError("sdk distribution failed")
        self.distribution_calls.append(_call(name, value, unit, attributes))

    def gauge(
        self,
        name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        attributes: Optional[dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        if self.fail:
            raise RuntimeError("sdk gauge failed")
        self.gauge_calls.append(_call(name, value, unit, attributes))


class _FakeSentry(ModuleType):
    def __init__(
        self,
        *,
        fail_metrics: bool = False,
        fail_flush: bool = False,
    ) -> None:
        super().__init__("sentry_sdk")
        self.metrics = _FakeMetrics(fail=fail_metrics)
        self.fail_flush = fail_flush
        self.init_calls: list[dict[str, object]] = []
        self.flush_calls: list[float] = []

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(dict(kwargs))

    def flush(self, *, timeout: float) -> None:
        if self.fail_flush:
            raise RuntimeError("sdk flush failed")
        self.flush_calls.append(timeout)


@pytest.fixture(autouse=True)
def reset_metrics_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "sentry_sdk", raising=False)
    monkeypatch.setattr(sentry_metrics_runtime, "_enabled", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_configured", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_sentry_sdk", None)


def test_configure_metrics_disabled_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> ModuleType:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(runtime_importlib, "import_module", fail_import)
    settings = _settings(enabled=False)

    configure_metrics(settings)
    count("ce.disabled")
    distribution("ce.disabled.duration", 12.0)
    gauge("ce.disabled.current", 7)
    flush()

    assert "sentry_sdk" not in sys.modules


def test_configure_metrics_missing_sdk_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_sdk(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(runtime_importlib, "import_module", missing_sdk)
    settings = _settings(enabled=True)

    configure_metrics(settings)
    count("ce.missing", attributes={"result": "ok"})
    flush()


def test_sdk_failures_are_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSentry(fail_metrics=True, fail_flush=True)
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

    configure_metrics(_settings(enabled=True))
    count("ce.sdk.count", attributes={"result": "ok"})
    distribution("ce.sdk.duration", 1.5)
    gauge("ce.sdk.gauge", 1)
    flush()

    assert fake.metrics.count_calls == []
    assert fake.metrics.distribution_calls == []
    assert fake.metrics.gauge_calls == []
    assert fake.flush_calls == []


def test_configure_metrics_initializes_sentry_once_with_privacy_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    settings = _settings(enabled=True)

    configure_metrics(settings)
    configure_metrics(settings)
    count("ce.init", attributes={"result": "ok"})

    assert len(fake.init_calls) == 1
    call = fake.init_calls[0]
    assert call["dsn"] == "https://public@example.invalid/1"
    assert call["environment"] == "test"
    assert call["release"] == "potpie-cli@test"
    assert call["dist"] is None
    assert call["send_default_pii"] is False
    assert call["include_local_variables"] is False
    assert call["max_request_body_size"] == "never"
    assert callable(call["before_send"])
    assert callable(call["before_breadcrumb"])
    assert fake.metrics.count_calls == [
        _MetricCall("ce.init", 1, None, {"result": "ok"}),
    ]


def test_count_distribution_and_gauge_emit_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

    configure_metrics(_settings(enabled=True))
    count("ce.count", value=3, attributes={"result": "ok"})
    distribution(
        "ce.duration_ms",
        12.5,
        unit="millisecond",
        attributes={"result": "ok"},
    )
    gauge("ce.active", 4, attributes={"state": "done"})

    assert fake.metrics.count_calls == [
        _MetricCall("ce.count", 3, None, {"result": "ok"}),
    ]
    assert fake.metrics.distribution_calls == [
        _MetricCall("ce.duration_ms", 12.5, "millisecond", {"result": "ok"}),
    ]
    assert fake.metrics.gauge_calls == [
        _MetricCall("ce.active", 4, None, {"state": "done"}),
    ]


def test_flush_is_noop_when_disabled_and_calls_sdk_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)

    configure_metrics(_settings(enabled=False))
    flush()
    assert fake.flush_calls == []

    configure_metrics(_settings(enabled=True))
    flush()

    assert fake.flush_calls == [2.0]


def test_repeated_configuration_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSentry()
    import_calls = 0

    def import_fake(name: str) -> ModuleType:
        nonlocal import_calls
        import_calls += 1
        assert name == "sentry_sdk"
        return fake

    monkeypatch.setattr(runtime_importlib, "import_module", import_fake)
    settings = _settings(enabled=True)

    configure_metrics(settings)
    configure_metrics(settings)
    count("ce.once", attributes={"result": "ok"})

    assert import_calls == 1
    assert fake.metrics.count_calls == [
        _MetricCall("ce.once", 1, None, {"result": "ok"}),
    ]


def _settings(*, enabled: bool) -> SentrySettings:
    return SentrySettings(
        enabled=enabled,
        dsn="https://public@example.invalid/1",
        environment="test",
        release="potpie-cli@test",
        dist=None,
    )


def _call(
    name: str,
    value: Union[int, float],
    unit: Optional[str],
    attributes: Optional[dict[str, Union[str, int, float, bool]]],
) -> _MetricCall:
    return _MetricCall(
        name=name,
        value=value,
        unit=unit,
        attributes={} if attributes is None else dict(attributes),
    )
