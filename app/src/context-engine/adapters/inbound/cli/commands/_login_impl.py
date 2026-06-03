"""Potpie-account login/logout implementations (host-routed CLI).

The real ``potpie login`` / ``potpie logout`` flows: a browser Firebase sign-in
(custom token → Firebase session, stored in the system keychain) or a direct
``--api-key`` store. Lifted out of the old monolithic ``main.py`` into its own
module so the heavy auth import tree (httpx, webbrowser, keyring) stays off
``build_app``'s eager import path — ``commands/auth.py`` imports these lazily
inside the command bodies.
"""

from __future__ import annotations

from datetime import datetime, timezone

import typer

from adapters.inbound.cli.auth.firebase_session import (
    FirebaseSessionError,
    exchange_custom_token,
)
from adapters.inbound.cli.auth.potpie import (
    PotpieCliAuthError,
    resolve_potpie_api_url_for_auth,
    revoke_api_key_on_server,
    run_browser_login_flow,
)
from adapters.inbound.cli.credentials_store import (
    CredentialStoreError,
    clear_potpie_auth,
    credentials_path,
    get_potpie_auth_type,
    get_stored_api_key,
    store_potpie_api_key,
    store_potpie_firebase_id_token,
    store_potpie_firebase_refresh_token,
    write_api_base_url,
)
from adapters.inbound.cli.output import emit_error, print_json_blob, print_plain_line


def _flags() -> tuple[bool, bool]:
    from adapters.inbound.cli.commands._common import is_json, is_verbose

    return is_json(), is_verbose()


def potpie_login_impl() -> None:
    """Browser sign-in: exchange a custom token and store the Firebase session."""
    j, v = _flags()
    try:
        print_plain_line(
            "Opening browser to sign in...\nWaiting for authentication...",
            as_json=False,
        )
        result = run_browser_login_flow()
        session = exchange_custom_token(
            result.custom_token,
            firebase_api_key=result.firebase_api_key,
        )
        store_potpie_firebase_refresh_token(
            session.refresh_token,
            created_at=datetime.now(timezone.utc).isoformat(),
            firebase_api_key=result.firebase_api_key,
        )
        store_potpie_firebase_id_token(session.id_token)
    except (PotpieCliAuthError, FirebaseSessionError, CredentialStoreError) as exc:
        emit_error("Potpie login failed", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    if j:
        print_json_blob(
            {"ok": True, "auth_type": "potpie", "token_storage": "keychain"},
            as_json=True,
        )
        return
    print_plain_line("Logged in to Potpie successfully.", as_json=False)


def potpie_logout_impl() -> None:
    """Remove Potpie CLI auth from the system keychain (revoking API keys)."""
    j, v = _flags()
    api_key = ""
    clear_api_key = False
    try:
        clear_api_key = get_potpie_auth_type() == "api_key"
        if clear_api_key:
            api_key = get_stored_api_key()
    except CredentialStoreError as exc:
        emit_error("Potpie logout failed", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    if api_key:
        try:
            revoke_api_key_on_server(
                api_base_url=resolve_potpie_api_url_for_auth(),
                api_key=api_key,
            )
        except PotpieCliAuthError as exc:
            emit_error("Potpie logout failed", str(exc), verbose=v)
            raise typer.Exit(code=1) from exc

    try:
        clear_potpie_auth(clear_api_key=clear_api_key)
    except CredentialStoreError as exc:
        emit_error("Potpie logout failed", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    if j:
        print_json_blob({"ok": True}, as_json=True)
        return
    print_plain_line("Logged out of Potpie.", as_json=False)


def potpie_login_api_key_impl(token: str, url: str | None) -> None:
    """Store a Potpie API key (and optional base URL) in the keyring."""
    j, v = _flags()
    try:
        store_potpie_api_key(token, created_at=datetime.now(timezone.utc).isoformat())
        write_api_base_url(url)
    except CredentialStoreError as exc:
        emit_error("Potpie login failed", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    print_plain_line(
        f"Saved API key to keyring ({credentials_path()}).",
        as_json=j,
        json_payload={"ok": True, "auth_type": "api_key", "path": str(credentials_path())},
    )


__all__ = ["potpie_login_impl", "potpie_logout_impl", "potpie_login_api_key_impl"]
