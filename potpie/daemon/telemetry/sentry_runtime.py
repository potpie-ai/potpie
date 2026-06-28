from __future__ import annotations


def configure_daemon_sentry() -> None:
    try:
        from potpie.runtime.telemetry.sentry_metrics import configure_metrics
        from potpie.runtime.telemetry.sentry_settings import load_sentry_settings

        configure_metrics(load_sentry_settings())
    except Exception:  # noqa: BLE001
        return


def capture_unexpected_daemon_error(
    exc: BaseException,
    *,
    error_code: str,
    error_kind: str,
) -> None:
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_tag("service", "potpie-daemon")
            scope.set_tag("error.code", error_code)
            scope.set_tag("error.kind", error_kind)
            scope.set_tag("is_expected", "false")
            sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001
        return


__all__ = ["capture_unexpected_daemon_error", "configure_daemon_sentry"]
