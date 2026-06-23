from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Final, Optional, Union

from potpie.context_engine.adapters.inbound.cli.telemetry.sentry_privacy import (
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
)
from potpie.context_engine.bootstrap.sentry_settings import SentrySettings

_ALLOWED_ATTRIBUTE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "arch",
        "backend",
        "cli_version",
        "command",
        "dry_run",
        "error_code",
        "hard",
        "host_mode",
        "os",
        "output_mode",
        "result",
        "scan",
        "state",
        "step",
        "subcommand",
    },
)

_configured = False
_enabled = False
_sentry_sdk: ModuleType | None = None

_MetricValue = Union[int, float]
_SafeMetricAttribute = Union[str, int, float, bool]
_MetricAttribute = Union[str, int, float, bool, Sequence[str], Mapping[str, str], None]
_MetricAttributes = Mapping[str, _MetricAttribute]
_MetricTags = dict[str, _SafeMetricAttribute]
_SentryInit = Callable[..., object]
_SentryMetric = Callable[..., object]
_SentryFlush = Callable[..., object]


def configure_metrics(settings: SentrySettings) -> None:
    global _configured, _enabled, _sentry_sdk
    if _configured:
        return
    if not settings.enabled or settings.dsn is None:
        _enabled = False
        return
    sentry_sdk = _load_sentry_sdk()
    if sentry_sdk is None:
        _enabled = False
        return
    try:
        sentry_init = _get_sentry_init(sentry_sdk)
        if sentry_init is None:
            _enabled = False
            return
        _ = sentry_init(
            dsn=settings.dsn,
            environment=settings.environment,
            release=settings.release,
            dist=settings.dist,
            send_default_pii=False,
            include_local_variables=False,
            max_request_body_size="never",
            before_send=scrub_sentry_event,
            before_breadcrumb=scrub_sentry_breadcrumb,
        )
        _sentry_sdk = sentry_sdk
        _configured = True
        _enabled = True
    # Sentry SDK failures must never affect context-engine control flow.
    except Exception:  # noqa: BLE001
        _enabled = False


def metrics_configured() -> bool:
    return _configured


def count(
    name: str,
    value: _MetricValue = 1,
    *,
    unit: Optional[str] = None,
    attributes: Optional[_MetricAttributes] = None,
) -> None:
    if not _enabled or _sentry_sdk is None:
        return
    safe_attributes = _safe_attributes(attributes)
    try:
        metric = _get_metric(_sentry_sdk, "count")
        if metric is not None:
            _ = metric(name, value, unit, attributes=safe_attributes)
    # Sentry SDK failures must never affect context-engine control flow.
    except Exception:  # noqa: BLE001
        return


def distribution(
    name: str,
    value: _MetricValue,
    *,
    unit: Optional[str] = None,
    attributes: Optional[_MetricAttributes] = None,
) -> None:
    if not _enabled or _sentry_sdk is None:
        return
    safe_attributes = _safe_attributes(attributes)
    try:
        metric = _get_metric(_sentry_sdk, "distribution")
        if metric is not None:
            _ = metric(name, value, unit, attributes=safe_attributes)
    # Sentry SDK failures must never affect context-engine control flow.
    except Exception:  # noqa: BLE001
        return


def gauge(
    name: str,
    value: _MetricValue,
    *,
    unit: Optional[str] = None,
    attributes: Optional[_MetricAttributes] = None,
) -> None:
    if not _enabled or _sentry_sdk is None:
        return
    safe_attributes = _safe_attributes(attributes)
    try:
        metric = _get_metric(_sentry_sdk, "gauge")
        if metric is not None:
            _ = metric(name, value, unit, attributes=safe_attributes)
    # Sentry SDK failures must never affect context-engine control flow.
    except Exception:  # noqa: BLE001
        return


def flush(timeout: float = 2.0) -> None:
    if not _enabled or _sentry_sdk is None:
        return
    try:
        sentry_flush = _get_sentry_flush(_sentry_sdk)
        if sentry_flush is not None:
            _ = sentry_flush(timeout=timeout)
    # Sentry SDK failures must never affect context-engine control flow.
    except Exception:  # noqa: BLE001
        return


def _load_sentry_sdk() -> ModuleType | None:
    try:
        return importlib.import_module("sentry_sdk")
    # Importing the external SDK can run package code outside this project.
    except Exception:  # noqa: BLE001
        return None


def _safe_attributes(
    attributes: Optional[_MetricAttributes],
) -> Optional[_MetricTags]:
    if attributes is None:
        return None
    safe: _MetricTags = {}
    for key, value in attributes.items():
        if key not in _ALLOWED_ATTRIBUTE_KEYS:
            continue
        safe_value = _safe_attribute_value(value)
        if safe_value is not None:
            safe[key] = safe_value
    return safe or None


def _safe_attribute_value(
    value: _MetricAttribute,
) -> Optional[_SafeMetricAttribute]:
    if isinstance(value, str):
        if _is_path_like(value):
            return None
        return value
    if isinstance(value, (bool, int, float)):
        return value
    return None


def _is_path_like(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("./")
        or value.startswith("../")
        or "/" in value
        or "\\" in value
    )


def _get_sentry_init(sentry_sdk: ModuleType) -> Optional[_SentryInit]:
    sentry_init = getattr(sentry_sdk, "init", None)
    if callable(sentry_init):
        return sentry_init
    return None


def _get_sentry_flush(sentry_sdk: ModuleType) -> Optional[_SentryFlush]:
    sentry_flush = getattr(sentry_sdk, "flush", None)
    if callable(sentry_flush):
        return sentry_flush
    return None


def _get_metric(
    sentry_sdk: ModuleType,
    metric_name: str,
) -> Optional[_SentryMetric]:
    sentry_metrics = getattr(sentry_sdk, "metrics", None)
    metric = getattr(sentry_metrics, metric_name, None)
    if callable(metric):
        return metric
    return None
