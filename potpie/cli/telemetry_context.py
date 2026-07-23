from __future__ import annotations

from pathlib import Path

import typer

from potpie.cli.telemetry.context import (
    TelemetryContext,
    bind_telemetry_context,
    current_telemetry_context,
)
from potpie.cli.telemetry.identity_store import load_or_create_identity


def bind_cli_telemetry_context(
    ctx: typer.Context, *, json_output: bool
) -> TelemetryContext:
    return bind_telemetry_context(ctx, json_output=json_output)


def load_anonymous_install_id(home: Path | None = None) -> str:
    # Compatibility shim: identity now always lives in the global config dir.
    _ = home
    return load_or_create_identity().anonymous_install_id


__all__ = [
    "TelemetryContext",
    "bind_cli_telemetry_context",
    "current_telemetry_context",
    "load_anonymous_install_id",
]
