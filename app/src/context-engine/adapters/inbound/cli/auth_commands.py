"""Generic CLI auth command foundation."""

from __future__ import annotations

import typer

auth_app = typer.Typer(help="Authenticate CLI integrations.")


def register_provider_app(name: str, provider_app: typer.Typer) -> None:
    """Register a provider-specific auth sub-application."""
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("provider app name must be non-empty")
    auth_app.add_typer(provider_app, name=key)
