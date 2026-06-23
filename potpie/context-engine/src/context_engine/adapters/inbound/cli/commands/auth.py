"""Integration-auth command surface: provider logins + Potpie account auth.

This registrar bolts the credential-acquisition surface onto the host-routed
root app:

- ``potpie github|linear|jira|confluence …`` — integration OAuth/API-token flows;
- ``potpie status [--verify]`` — local integration auth status (see ``bootstrap``);
- top-level ``potpie login`` / ``potpie logout`` — Potpie account (Firebase/API key);
- ``potpie auth …`` — deprecated aliases for the provider commands above.

These flows are inbound-adapter credential acquisition (OAuth/device-flow/keyring),
so they do NOT route through ``HostShell``; they read the shared ``--json`` /
``--verbose`` state from ``commands/_common`` like every other command.
"""

from __future__ import annotations

import typer


def register(root: typer.Typer) -> None:
    from context_engine.adapters.inbound.cli.auth.auth_commands import register_integration_commands

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
        from context_engine.adapters.inbound.cli.auth._login_impl import (
            potpie_login_api_key_impl,
            potpie_login_impl,
        )

        if api_key is not None:
            potpie_login_api_key_impl(api_key, url)
        else:
            potpie_login_impl()

    @root.command("logout")
    def logout() -> None:
        """Remove Potpie account auth from the system keychain."""
        from context_engine.adapters.inbound.cli.auth._login_impl import potpie_logout_impl

        potpie_logout_impl()


__all__ = ["register"]
