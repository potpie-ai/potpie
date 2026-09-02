from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Final, Optional, Union

from potpie.cli.telemetry.sentry_privacy import (
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
)
from potpie_context_engine.bootstrap.sentry_settings import SentrySettings

_ALLOWED_ATTRIBUTE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "arch",
        "backend",
        "backend_profile",
        "backend_ready",
        "cli_version",
        "command",
        "dry_run",
        "error_code",
        "hard",
        "host_mode",
        "match_mode",
        "operation",
        "os",
        "output_mode",
        "report",
        "result",
        "risk",
        "scan",
        "state",
        "step",
        "status",
        "subgraph",
        "subcommand",
        "view",
    },
)

_configured = False
_enabled = False
_sentry_sdk: ModuleType | None = None

#: How long a short-lived process may wait at exit for its one metrics
#: envelope. The SDK default is 2 s and it is paid on every command that had
#: something to send; a CLI that runs for 0.3 s cannot spend that on telemetry.
_SHORT_LIVED_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 1.0

_MetricValue = Union[int, float]
_SafeMetricAttribute = Union[str, int, float, bool]
_MetricAttribute = Union[str, int, float, bool, Sequence[str], Mapping[str, str], None]
_MetricAttributes = Mapping[str, _MetricAttribute]
_MetricTags = dict[str, _SafeMetricAttribute]
_SentryInit = Callable[..., object]
_SentryMetric = Callable[..., object]
_SentryFlush = Callable[..., object]


def configure_metrics(
    settings: SentrySettings, *, short_lived_process: bool = False
) -> None:
    """Initialise the SDK once.

    ``short_lived_process`` is the CLI profile. The SDK's default ``init`` probes
    forty auto-enabling integrations by importing their target packages
    (fastapi, starlette, redis, httpx, aiohttp, huggingface_hub, …), which on a
    fully installed potpie costs ~1,250 extra modules and close to half a second
    of CPU per command — for integrations a one-shot process never uses. The
    profile turns the probes off, bounds the exit flush, and silences the
    SDK's atexit notice ("Sentry is attempting to send N pending events"),
    which would otherwise print to stderr on every command now that nothing
    flushes synchronously before exit. Long-lived processes (daemon, workers)
    keep the SDK defaults.
    """
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
        init_options: dict[str, object] = {
            "dsn": settings.dsn,
            "environment": settings.environment,
            "release": settings.release,
            "dist": settings.dist,
            "send_default_pii": False,
            "include_local_variables": False,
            "max_request_body_size": "never",
            "before_send": scrub_sentry_event,
            "before_breadcrumb": scrub_sentry_breadcrumb,
        }
        if short_lived_process:
            init_options.update(_short_lived_process_options())
        _ = sentry_init(**init_options)
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


def _short_lived_process_options() -> dict[str, object]:
    options: dict[str, object] = {
        "auto_enabling_integrations": False,
        "shutdown_timeout": _SHORT_LIVED_SHUTDOWN_TIMEOUT_SECONDS,
    }
    quiet_atexit = _quiet_atexit_integration()
    if quiet_atexit is not None:
        options["integrations"] = [quiet_atexit]
    return options


def _quiet_atexit_integration() -> object | None:
    """The SDK's atexit flush with its stderr notice removed, or ``None``.

    Passing an explicit ``AtexitIntegration`` replaces the default one of the
    same name; the default's callback writes "Sentry is attempting to send …"
    to stderr whenever an envelope is still pending at exit, which for a CLI
    that defers its only flush to exit means on every command.
    """
    try:
        atexit_module = importlib.import_module("sentry_sdk.integrations.atexit")
        integration = getattr(atexit_module, "AtexitIntegration", None)
        if integration is None:
            return None
        return integration(callback=_silent_shutdown_callback)
    # A missing or reshaped integration module only costs the silence.
    except Exception:  # noqa: BLE001
        return None


def _silent_shutdown_callback(pending: int, timeout: float) -> None:
    del pending, timeout


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
