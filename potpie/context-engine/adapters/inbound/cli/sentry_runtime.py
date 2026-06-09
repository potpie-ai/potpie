from __future__ import annotations

from adapters.inbound.cli.telemetry_context import current_telemetry_context


def configure_cli_sentry() -> None:
    try:
        from observability import configure, profiles

        configure(profiles.cli())
    except Exception:  # noqa: BLE001
        return


def capture_unexpected_cli_error(
    exc: BaseException,
    *,
    error_code: str,
    error_kind: str,
) -> None:
    try:
        import sentry_sdk

        telemetry = current_telemetry_context()
        fields = telemetry.fields() if telemetry is not None else {}
        fields.update(
            {
                "service": "potpie-cli",
                "error.code": error_code,
                "error.kind": error_kind,
                "is_expected": "false",
            }
        )
        with sentry_sdk.new_scope() as scope:
            for key in (
                "service",
                "cli_version",
                "python_version",
                "os",
                "arch",
                "command",
                "subcommand",
                "output_mode",
                "error.code",
                "error.kind",
                "is_expected",
            ):
                value = fields.get(key)
                if value is not None:
                    scope.set_tag(key, str(value))
            scope.set_context(
                "telemetry",
                {
                    key: str(fields[key])
                    for key in (
                        "anonymous_install_id",
                        "invocation_id",
                        "daemon_session_id",
                    )
                    if key in fields
                },
            )
            sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001
        return
