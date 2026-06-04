"""Integration-auth command surface: ``login``/``logout`` + ``auth`` group.

This registrar bolts the credential-acquisition surface onto the host-routed
root app:

- top-level ``potpie login`` / ``potpie logout`` — the real Potpie-account flow
  (browser Firebase sign-in or ``--api-key``), implemented in ``_login_impl``;
- ``potpie auth …`` — the integration logins (``github``/``linear``/``jira``/
  ``confluence``) assembled in ``auth_commands`` + ``auth/github_commands``.

These flows are inbound-adapter credential acquisition (OAuth/device-flow/keyring),
so they do NOT route through ``HostShell``; they read the shared ``--json`` /
``--verbose`` state from ``commands/_common`` like every other command. Heavy
imports (httpx, webbrowser, keyring) stay inside ``register``/command bodies so
``potpie --help`` and unrelated commands stay fast.
"""

from __future__ import annotations

import typer


def register(root: typer.Typer) -> None:
    # Importing auth_commands self-registers the linear/jira/confluence sub-apps
    # into auth_app at module load; register_github_commands then attaches github
    # into auth_app (and github/git at root). Both must run BEFORE auth_app is
    # mounted under `auth` so the sub-apps are present.
    from adapters.inbound.cli.auth.github_commands import register_github_commands
    from adapters.inbound.cli.auth.auth_commands import auth_app

    register_github_commands(root)
    root.add_typer(auth_app, name="auth")

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
        from adapters.inbound.cli.auth._login_impl import (
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
        from adapters.inbound.cli.auth._login_impl import potpie_logout_impl

        potpie_logout_impl()


__all__ = ["register"]
