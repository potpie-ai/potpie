from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Optional, Union

import pytest
from potpie.runtime.telemetry.sentry_settings import SentrySettings

sentry_metrics_runtime = import_module("potpie.runtime.telemetry.sentry_metrics")
configure_metrics = getattr(sentry_metrics_runtime, "configure_metrics")
count = getattr(sentry_metrics_runtime, "count")


@dataclass(frozen=True)
class _MetricCall:
    name: str
    value: Union[int, float]
    unit: Optional[str]
    attributes: Mapping[str, Union[str, int, float, bool]]


class _FakeMetrics:
    def __init__(self) -> None:
        self.count_calls: list[_MetricCall] = []

    def count(
        self,
        name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        attributes: Optional[dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        self.count_calls.append(_call(name, value, unit, attributes))


class _FakeSentry(ModuleType):
    def __init__(self) -> None:
        super().__init__("sentry_sdk")
        self.metrics = _FakeMetrics()
        self.init_calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(dict(kwargs))


@pytest.fixture(autouse=True)
def reset_metrics_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "sentry_sdk", raising=False)
    monkeypatch.setattr(sentry_metrics_runtime, "_enabled", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_configured", False)
    monkeypatch.setattr(sentry_metrics_runtime, "_sentry_sdk", None)


def test_runtime_helper_stays_plain_functions_without_protocol_abstractions() -> None:
    source = inspect.getsource(sentry_metrics_runtime)
    tree = ast.parse(source)

    class_defs = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]

    assert class_defs == []
    assert "Proto" + "col" not in source
    assert "runtime" + "_checkable" not in source
    assert "Metric" + "Sink" not in source


def test_count_keeps_only_low_cardinality_allowlisted_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    expected_attrs: dict[str, Union[str, int, float, bool]] = {
        "command": "setup",
        "subcommand": "run",
        "output_mode": "json",
        "cli_version": "0.1.0",
        "os": "darwin",
        "arch": "arm64",
        "result": "ok",
        "error_code": "none",
        "backend": "falkordb",
        "backend_profile": "falkordb_lite",
        "backend_ready": True,
        "host_mode": "daemon",
        "match_mode": "vector",
        "operation": "add",
        "report": "summary",
        "risk": "low",
        "scan": "repo",
        "dry_run": False,
        "step": "workspace",
        "state": "done",
        "status": "validated",
        "subgraph": "recent_changes",
        "view": "recent_changes.timeline",
        "hard": True,
    }

    configure_metrics(_settings(enabled=True))
    count(
        "ce.cli.invocations_total",
        attributes={
            **expected_attrs,
            "pot_id": "pot-1",
            "source_channel": "github",
            "source_system": "pull_request",
            "event_type": "comment",
            "action": "created",
            "ingestion_kind": "diff_sync",
        },
    )

    assert fake.metrics.count_calls == [
        _MetricCall("ce.cli.invocations_total", 1, None, expected_attrs),
    ]


def test_count_drops_high_cardinality_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    attrs = {
        "command": "scan",
        "result": "ok",
        "dry_run": True,
        "state": "src/main.py",
        "nested": {"result": "ok"},
        "items": ["one", "two"],
        "none": None,
    }
    banned_keys = (
        "pot_id",
        "event_id",
        "batch_id",
        "invocation_id",
        "daemon_session_id",
        "anonymous_install_id",
        "repo_name",
        "source_id",
        "file_path",
        "path",
        "prompt",
        "exception_message",
        "raw_exception_message",
        "auth_subject",
        "user_id",
        "org_id",
        "token",
        "access_token",
        "refresh_token",
    )
    attrs.update(dict.fromkeys(banned_keys, "sensitive"))

    configure_metrics(_settings(enabled=True))
    count("ce.test", attributes=attrs)

    safe_attrs = {"command": "scan", "result": "ok", "dry_run": True}
    assert fake.metrics.count_calls == [_MetricCall("ce.test", 1, None, safe_attrs)]


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
