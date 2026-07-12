"""Graph CLI command envelope, telemetry, and result emission helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import logging
from typing import Any

import typer
from potpie.runtime.graph_compat import (
    SPAN_KIND_INTERNAL,
    GraphUnsupported,
    get_observability,
    graph_error_envelope,
    graph_not_implemented_envelope,
    graph_success_envelope,
    new_graph_request_id,
    normalize_workbench_result,
)

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    EXIT_OPERATION,
    contract,
    emit,
    is_json,
    json_error_formatter,
)
from potpie.cli.commands.graph_read import (
    _effective_read_format,
    _emit_read,
    _raw_item_rows,
    _read_human,
    _read_payload,
    _timeline_events,
    _with_read_context,
)
from potpie.cli.commands.graph_render import _inbox_human, _quality_human
from potpie.cli.telemetry.product_analytics import AnalyticsValue
from potpie.cli.telemetry.usage_events import capture_usage_command_succeeded

_LOG = logging.getLogger(__name__)

_GRAPH_USAGE_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
        "backend_profile",
        "backend_ready",
        "match_mode",
        "operation",
        "report",
        "risk",
        "status",
        "subgraph",
        "view",
    }
)


class _GraphCliCommandContext:
    def __init__(self, command: str) -> None:
        self.command = command
        self.request_id = new_graph_request_id()
        self.pot_id: str | None = None
        self.subgraph_versions: dict[str, int] = {}
        self.telemetry_result = "ok"
        self.telemetry_error_code = "none"
        self.telemetry_attributes: dict[str, str] = {}

    def set_pot_id(self, pot_id: str | None) -> None:
        self.pot_id = pot_id

    def set_subgraph_versions(self, versions: Mapping[str, Any] | None) -> None:
        if not versions:
            return
        clean: dict[str, int] = {}
        for key, value in versions.items():
            try:
                clean[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        self.subgraph_versions = clean

    def format_error(self, payload: dict[str, Any]) -> dict[str, Any]:
        error = payload.get("error")
        code = str(error.get("code") if isinstance(error, dict) else "error")
        self.mark_result(result=code, error_code=code)
        meta = payload.get("meta")
        if isinstance(meta, dict):
            meta["command"] = self.command
            meta["request_id"] = self.request_id
        return payload

    def mark_result(
        self,
        *,
        result: str,
        error_code: str = "none",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.telemetry_result = result
        self.telemetry_error_code = error_code
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                self.telemetry_attributes[str(key)] = str(value)


@contextmanager
def _graph_command(command: str) -> Iterator[_GraphCliCommandContext]:
    ctx = _GraphCliCommandContext(command)
    obs = get_observability()
    span_name, base_attrs = _graph_telemetry_shape(command)
    started_at = time.perf_counter()
    with obs.span(
        span_name,
        kind=SPAN_KIND_INTERNAL,
        attributes={
            **base_attrs,
            "command": command,
            "request_id": ctx.request_id,
        },
    ) as span:
        try:
            with json_error_formatter(ctx.format_error):
                with contract():
                    yield ctx
        except BaseException as exc:
            if ctx.telemetry_error_code == "none":
                if isinstance(exc, typer.Exit):
                    result = "ok" if (exc.exit_code in (None, 0)) else "exit"
                    ctx.mark_result(result=result, error_code="exit")
                else:
                    ctx.mark_result(
                        result="unexpected",
                        error_code=exc.__class__.__name__,
                    )
                    span.record_exception(exc)
                    span.set_error(repr(exc))
            raise
        finally:
            duration_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
            attrs = {
                **base_attrs,
                **ctx.telemetry_attributes,
                "command": command,
                "request_id": ctx.request_id,
                "result": ctx.telemetry_result,
                "error_code": ctx.telemetry_error_code,
            }
            if ctx.pot_id:
                attrs["pot_id"] = ctx.pot_id
            span.set_attributes(attrs)
            if ctx.telemetry_error_code != "none":
                span.set_error(ctx.telemetry_error_code)
            _record_graph_command_telemetry(
                obs,
                command=command,
                duration_ms=duration_ms,
                attributes=attrs,
            )
            _record_graph_command_usage_event(
                command=command,
                duration_ms=duration_ms,
                result=ctx.telemetry_result,
                error_code=ctx.telemetry_error_code,
                attributes=attrs,
            )


def _graph_telemetry_shape(command: str) -> tuple[str, dict[str, str]]:
    raw = command.removeprefix("graph.").replace("-", "_")
    parts = raw.split(".")
    if not parts:
        return "graph.unknown", {}
    if parts[0] == "inbox":
        attrs = {"operation": parts[1] if len(parts) > 1 else "unknown"}
        return "graph.inbox", attrs
    if parts[0] == "quality":
        attrs = {"report": parts[1] if len(parts) > 1 else "summary"}
        return "graph.quality", attrs
    return f"graph.{parts[0]}", {}


def _record_graph_command_telemetry(
    obs: Any,
    *,
    command: str,
    duration_ms: float,
    attributes: Mapping[str, str],
) -> None:
    _span_name, base_attrs = _graph_telemetry_shape(command)
    raw = command.removeprefix("graph.").replace("-", "_")
    root = raw.split(".", 1)[0] if raw else "unknown"
    metric_root = root
    metric_attrs = dict(base_attrs)
    metric_attrs.update(
        {
            key: value
            for key, value in attributes.items()
            if key
            in {
                "result",
                "error_code",
                "pot_id",
                "subgraph",
                "view",
                "risk",
                "status",
                "operation",
                "report",
                "backend_profile",
                "match_mode",
            }
        }
    )
    try:
        obs.counter(f"ce.graph.{metric_root}_total", 1, attributes=metric_attrs)
        obs.histogram(
            f"ce.graph.{metric_root}_ms",
            duration_ms,
            attributes=metric_attrs,
        )
    except Exception:  # noqa: BLE001 - observability must never fail a command
        _LOG.debug("graph command observability metric emission failed", exc_info=True)
    try:
        from potpie.runtime.telemetry import sentry_metrics

        sentry_metrics.count(
            f"ce.graph.{metric_root}_total",
            attributes=metric_attrs,
        )
        sentry_metrics.distribution(
            f"ce.graph.{metric_root}_ms",
            duration_ms,
            unit="millisecond",
            attributes=metric_attrs,
        )
        sentry_metrics.flush(timeout=2.0)
    except Exception:  # noqa: BLE001 - Sentry metrics must never fail a command
        _LOG.debug("graph command Sentry metric emission failed", exc_info=True)


def _record_graph_command_usage_event(
    *,
    command: str,
    duration_ms: float,
    result: str,
    error_code: str,
    attributes: Mapping[str, Any],
) -> None:
    if result != "ok" or error_code != "none":
        return
    props: dict[str, AnalyticsValue] = {
        "command_family": _graph_command_family(command),
        "duration_ms": round(duration_ms, 3),
        "error_code": error_code,
        "result": result,
    }
    for key in sorted(_GRAPH_USAGE_ATTRIBUTE_KEYS):
        value = attributes.get(key)
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            props[key] = value
        else:
            props[key] = str(value)
    capture_usage_command_succeeded(
        command=command,
        result_kind="graph_command",
        properties=props,
    )


def _graph_command_family(command: str) -> str:
    raw = command.removeprefix("graph.").replace("-", "_")
    return raw.split(".", 1)[0] if raw else "unknown"


def _graph_telemetry_attributes(result: Mapping[str, Any]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key in (
        "subgraph",
        "view",
        "risk",
        "status",
        "report",
        "action",
        "backend_profile",
        "match_mode",
    ):
        value = result.get(key)
        if value is not None:
            target = "operation" if key == "action" else key
            attrs[target] = str(value)
    backend = result.get("backend")
    if isinstance(backend, Mapping):
        if backend.get("profile") is not None:
            attrs["backend_profile"] = str(backend["profile"])
        if backend.get("ready") is not None:
            attrs["backend_ready"] = str(bool(backend["ready"])).lower()
    graph_service = result.get("graph_service")
    if (
        isinstance(graph_service, Mapping)
        and graph_service.get("match_mode") is not None
    ):
        attrs["match_mode"] = str(graph_service["match_mode"])
    return attrs


def _emit_graph_result(
    ctx: _GraphCliCommandContext,
    payload: Mapping[str, Any],
    *,
    human: str,
    warnings: tuple[str, ...] = (),
    unsupported: tuple[GraphUnsupported, ...] = (),
    recommended_next_action: str | None = None,
) -> None:
    result, versions, payload_warnings, payload_unsupported = (
        normalize_workbench_result(payload)
    )
    if versions:
        ctx.set_subgraph_versions(versions)
    merged_warnings = tuple(warnings) + payload_warnings
    merged_unsupported = tuple(unsupported) + payload_unsupported
    result_label = "ok"
    error_code = "none"
    if payload.get("ok", True) is False:
        result_label = str(payload.get("status") or _error_code_from_result(payload))
        error_code = _error_code_from_result(payload)
    ctx.mark_result(
        result=result_label,
        error_code=error_code,
        attributes=_graph_telemetry_attributes(result),
    )

    if payload.get("ok", True) is False:
        env = graph_error_envelope(
            command=ctx.command,
            request_id=ctx.request_id,
            pot_id=ctx.pot_id,
            code=_error_code_from_result(payload),
            message=_error_message_from_result(payload),
            detail=result or None,
            subgraph_versions=ctx.subgraph_versions,
            warnings=merged_warnings,
            unsupported=merged_unsupported,
            recommended_next_action=recommended_next_action
            or payload.get("recommended_next_action"),
        )
    else:
        env = graph_success_envelope(
            command=ctx.command,
            request_id=ctx.request_id,
            pot_id=ctx.pot_id,
            result=result,
            subgraph_versions=ctx.subgraph_versions,
            warnings=merged_warnings,
            unsupported=merged_unsupported,
            recommended_next_action=recommended_next_action
            or payload.get("recommended_next_action"),
        )
    emit(env.to_dict(), human=_with_graph_warnings(human, merged_warnings))


def _with_graph_warnings(human: str, warnings: tuple[str, ...]) -> str:
    if not warnings:
        return human
    return "\n".join([human, *(f"! {warning}" for warning in warnings)])


def _emit_inbox_result(ctx: _GraphCliCommandContext, result: Any) -> None:
    _emit_graph_result(ctx, result.to_dict(), human=_inbox_human(result))
    if not result.ok:
        raise typer.Exit(code=EXIT_OPERATION)


def _emit_quality_result(ctx: _GraphCliCommandContext, result: Any) -> None:
    _emit_graph_result(ctx, result.to_dict(), human=_quality_human(result))
    if not result.ok:
        raise typer.Exit(code=EXIT_OPERATION)


def _emit_graph_not_implemented(
    ctx: _GraphCliCommandContext,
    *,
    detail: str | None = None,
    recommended_next_action: str | None = None,
) -> None:
    env = graph_not_implemented_envelope(
        command=ctx.command,
        request_id=ctx.request_id,
        pot_id=ctx.pot_id,
        detail=detail,
        recommended_next_action=recommended_next_action,
    )
    emit(env.to_dict(), human=detail or f"{ctx.command} is not implemented yet")
    raise typer.Exit(code=EXIT_UNAVAILABLE)


def _error_code_from_result(payload: Mapping[str, Any]) -> str:
    issues = payload.get("issues")
    if isinstance(issues, list) and issues:
        first = issues[0]
        if isinstance(first, Mapping) and first.get("code"):
            return str(first["code"])
    return str(payload.get("status") or "graph_command_failed")


def _error_message_from_result(payload: Mapping[str, Any]) -> str:
    if payload.get("detail"):
        return str(payload["detail"])
    issues = payload.get("issues")
    if isinstance(issues, list) and issues:
        first = issues[0]
        if isinstance(first, Mapping) and first.get("message"):
            return str(first["message"])
    status = payload.get("status")
    return f"Graph command failed with status {status!r}."


def _emit_graph_read(
    ctx: _GraphCliCommandContext,
    result: Any,
    *,
    format_: str,
    sort: str,
    dedupe: str,
    event_limit: int | None = None,
    human_prefix: str | None = None,
    warnings: tuple[str, ...] = (),
) -> None:
    if not is_json():
        _emit_read(
            result,
            format_=format_,
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
            human_prefix=human_prefix,
            warnings=warnings,
        )
        if result.to_dict().get("ok", True) is False:
            raise typer.Exit(code=EXIT_OPERATION)
        return

    normalized_format = _effective_read_format(result, format_)
    if normalized_format == "jsonl":
        rows = _timeline_events(result, sort=sort, dedupe=dedupe, limit=event_limit)
        if not rows:
            rows = _raw_item_rows(result)
        payload = _read_payload(
            result,
            format_="raw",
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        )
        if payload.get("detail") != "full":
            payload.pop("items", None)
        payload["read_shape"] = "jsonl"
        payload["rows"] = rows
        payload["row_count"] = len(rows)
    else:
        payload = _read_payload(
            result,
            format_=normalized_format,
            sort=sort,
            dedupe=dedupe,
            event_limit=event_limit,
        )
    failed = payload.get("ok", True) is False
    _emit_graph_result(
        ctx,
        payload,
        human=_with_read_context(
            _read_human(
                result,
                format_=normalized_format,
                sort=sort,
                dedupe=dedupe,
                event_limit=event_limit,
            ),
            human_prefix=human_prefix,
            warnings=warnings,
        ),
        warnings=warnings,
    )
    if failed:
        raise typer.Exit(code=EXIT_OPERATION)
