"""Integration-auth command surface: provider logins + Potpie account auth.

This registrar bolts the credential-acquisition surface onto the runtime-routed
root app:

- ``potpie integration <provider> …`` — integration OAuth/API-token flows;
- ``potpie integration status [PROVIDER]`` — local integration auth status;
- top-level ``potpie login`` / ``potpie logout`` — Potpie account (Firebase/API key);

These flows are inbound-adapter credential acquisition
(OAuth/device-flow/local files), so they do not cross engine RPC; they read the
shared ``--json`` / ``--verbose`` state from ``commands/_common`` like every
other command.
"""

from __future__ import annotations

import typer


def register(root: typer.Typer) -> None:
    from potpie.auth.auth_commands import register_integration_commands

    register_integration_commands(root)

    @root.command("login")
    def login(
        api_key: str = typer.Option(
            None,
            "--api-key",
            "-k",
            help="Potpie API key (sk-…). Uses key auth instead of browser login.",
        ),
        url: str = typer.Option(
            None,
            "--url",
            "-u",
            help="Potpie API base URL (only with --api-key), e.g. http://127.0.0.1:8001",
        ),
    ) -> None:
        """Sign in to Potpie (browser Firebase session, or --api-key to store a key)."""
        from potpie.auth._login_impl import (
            potpie_login_api_key_impl,
            potpie_login_impl,
        )

        if api_key is not None:
            potpie_login_api_key_impl(api_key, url)
        else:
            potpie_login_impl()

    @root.command("logout")
    def logout() -> None:
        """Remove Potpie account auth from local credential files."""
        from potpie.auth._login_impl import potpie_logout_impl

        potpie_logout_impl()


__all__ = ["register"]
