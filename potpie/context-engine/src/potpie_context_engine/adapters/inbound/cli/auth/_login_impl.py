"""Potpie-account login/logout implementations (host-routed CLI).

The real ``potpie login`` / ``potpie logout`` flows: a browser Firebase sign-in
(custom token → Firebase session, stored in the local credentials file) or a direct
``--api-key`` store. Lifted out of the old monolithic ``main.py`` into its own
module so the heavy auth import tree (httpx, webbrowser) stays off
``build_app``'s eager import path — ``commands/auth.py`` imports these lazily
inside the command bodies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NoReturn

import typer

from potpie_context_engine.adapters.outbound.cli_auth.firebase_session import (
    FirebaseSessionError,
    exchange_custom_token,
)
from potpie_context_engine.adapters.outbound.cli_auth.potpie import (
    PotpieCliAuthError,
    resolve_potpie_api_url_for_auth,
    revoke_api_key_on_server,
    run_browser_login_flow,
)
from potpie_context_engine.adapters.outbound.cli_auth.credentials_store import CredentialStoreError
from potpie_context_engine.adapters.inbound.cli.commands._common import EXIT_AUTH, get_store
from potpie_context_engine.adapters.inbound.cli.ui.output import emit_error, print_json_blob, print_plain_line


def _flags() -> tuple[bool, bool]:
    from potpie_context_engine.adapters.inbound.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def potpie_login_impl() -> None:
    """Browser sign-in: exchange a custom token and store the Firebase session."""
    j, v = _flags()
    try:
        store = get_store()
        print_plain_line(
            "Opening browser to sign in...\nWaiting for authentication...",
            as_json=False,
        )
        result = run_browser_login_flow()
        session = exchange_custom_token(
            result.custom_token,
            firebase_api_key=result.firebase_api_key,
        )
        store.store_potpie_firebase_refresh_token(
            session.refresh_token,
            created_at=datetime.now(timezone.utc).isoformat(),
            firebase_api_key=result.firebase_api_key,
        )
        store.store_potpie_firebase_id_token(session.id_token)
    except (PotpieCliAuthError, FirebaseSessionError, CredentialStoreError) as exc:
        emit_error("Potpie login failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_potpie_auth_error(
            exc, title="Potpie login failed", verbose=v
        )

    if j:
        print_json_blob(
            {"ok": True, "auth_type": "potpie", "token_storage": "file"},
            as_json=True,
        )
        return
    print_plain_line("Logged in to Potpie successfully.", as_json=False)


def potpie_logout_impl() -> None:
    """Remove Potpie CLI auth from local credential files (revoking API keys)."""
    j, v = _flags()
    api_key = ""
    clear_api_key = False
    try:
        store = get_store()
        clear_api_key = store.get_potpie_auth_type() == "api_key"
        if clear_api_key:
            api_key = store.get_stored_api_key()
    except CredentialStoreError as exc:
        emit_error("Potpie logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_potpie_auth_error(
            exc, title="Potpie logout failed", verbose=v
        )

    if api_key:
        try:
            revoke_api_key_on_server(
                api_base_url=resolve_potpie_api_url_for_auth(),
                api_key=api_key,
            )
        except PotpieCliAuthError as exc:
            emit_error("Potpie logout failed", str(exc), verbose=v)
            raise typer.Exit(code=EXIT_AUTH) from exc

    try:
        store.clear_potpie_auth(clear_api_key=clear_api_key)
    except CredentialStoreError as exc:
        emit_error("Potpie logout failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_potpie_auth_error(
            exc, title="Potpie logout failed", verbose=v
        )

    if j:
        print_json_blob({"ok": True}, as_json=True)
        return
    print_plain_line("Logged out of Potpie.", as_json=False)


def potpie_login_api_key_impl(token: str, url: str | None) -> None:
    """Store a Potpie API key (and optional base URL) in the local credentials file."""
    j, v = _flags()
    try:
        store = get_store()
        store.store_potpie_api_key(
            token, created_at=datetime.now(timezone.utc).isoformat()
        )
        store.write_api_base_url(url)
    except CredentialStoreError as exc:
        emit_error("Potpie login failed", str(exc), verbose=v)
        raise typer.Exit(code=EXIT_AUTH) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _capture_unexpected_potpie_auth_error(
            exc, title="Potpie login failed", verbose=v
        )
    path = store.credentials_path()
    print_plain_line(
        f"Saved API key to local credentials file ({path}).",
        as_json=j,
        json_payload={
            "ok": True,
            "auth_type": "api_key",
            "token_storage": "file",
            "path": str(path),
        },
    )


def _capture_unexpected_potpie_auth_error(
    exc: BaseException,
    *,
    title: str,
    verbose: bool,
) -> NoReturn:
    from potpie_context_engine.adapters.inbound.cli.telemetry.sentry_runtime import (
        capture_unexpected_cli_error,
    )

    capture_unexpected_cli_error(
        exc,
        error_code="unexpected_cli_error",
        error_kind="unexpected",
    )
    emit_error(
        title,
        "Unexpected internal error.",
        code="unexpected_cli_error",
        verbose=verbose,
    )
    raise typer.Exit(code=EXIT_AUTH) from exc


__all__ = ["potpie_login_impl", "potpie_logout_impl", "potpie_login_api_key_impl"]
