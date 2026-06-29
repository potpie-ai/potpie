from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import EXIT_UNAVAILABLE, emit, fail
from adapters.inbound.cli.telemetry import sentry_runtime, settings
from adapters.inbound.cli.telemetry.preferences import (
    TelemetryPreferenceWriteError,
    TelemetryPreferences,
    save_preferences,
)
from adapters.inbound.cli.telemetry.product_analytics import configure_product_analytics

telemetry_app = typer.Typer(help="CLI telemetry controls.")


@telemetry_app.command("status")
def status() -> None:
    """Show Potpie CLI telemetry status."""
    _emit_status()


@telemetry_app.command("enable")
def enable() -> None:
    """Enable anonymous Potpie CLI telemetry."""
    _save_preferences(TelemetryPreferences(enabled=True))
    _refresh_runtime_sinks()
    _emit_status()


@telemetry_app.command("disable")
def disable() -> None:
    """Disable outbound Potpie CLI telemetry."""
    _save_preferences(TelemetryPreferences(enabled=False))
    _refresh_runtime_sinks()
    _emit_status()


def _emit_status() -> None:
    status = settings.load_telemetry_status()
    payload = {
        "telemetry": status.telemetry,
        "crash_reports": status.crash_reports,
        "analytics": status.analytics,
    }
    emit(payload, human=_human_status(payload))


def _save_preferences(preferences: TelemetryPreferences) -> None:
    try:
        save_preferences(preferences)
    except TelemetryPreferenceWriteError as exc:
        fail(
            code="telemetry_preference_write_failed",
            message="Could not update telemetry preference.",
            detail=str(exc),
            next_action="check that the Potpie config directory is writable, then retry",
            exit_code=EXIT_UNAVAILABLE,
        )


def _refresh_runtime_sinks() -> None:
    sentry_runtime.configure_cli_sentry(settings.load_sentry_settings())
    configure_product_analytics(settings.load_product_analytics_settings())


def _human_status(payload: dict[str, str]) -> str:
    lines = [f"Potpie CLI telemetry: {payload['telemetry']}"]
    if payload["telemetry"] in {"blocked", "disabled"}:
        return "\n".join(lines)
    lines.extend(
        [
            "",
            f"Crash reports: {payload['crash_reports']}",
            f"Analytics: {payload['analytics']}",
        ]
    )
    return "\n".join(lines)


__all__ = ["telemetry_app"]
