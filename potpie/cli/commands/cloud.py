"""Reserved cloud login, status, push, pull and skill-sync commands.

These commands return structured not-implemented errors. Managed-host routing
for engine operations and pot management is implemented separately through
``potpie host`` and the shared command groups.
"""

from __future__ import annotations

import typer

from potpie.cli.commands._common import contract
from potpie_context_engine.core.errors import CapabilityNotImplemented

cloud_app = typer.Typer(
    help="Managed Potpie profile and sync (in development).",
    invoke_without_command=True,
)
skills_app = typer.Typer(
    help="Managed skill catalog sync (in development).",
    invoke_without_command=True,
)
cloud_app.add_typer(skills_app, name="skills")


@cloud_app.callback()
def _cloud_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        with contract():
            _todo("cloud")


@skills_app.callback()
def _skills_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        with contract():
            _todo("skills")


def _todo(op: str) -> None:
    raise CapabilityNotImplemented(
        f"cloud.{op}",
        detail="managed profile not implemented",
        recommended_next_action="use the local profile; cloud routing is on the roadmap",
    )


@cloud_app.command("login")
def cloud_login() -> None:
    with contract():
        _todo("login")


@cloud_app.command("status")
def cloud_status() -> None:
    with contract():
        _todo("status")


@cloud_app.command("push")
def cloud_push(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        _todo("push")


@cloud_app.command("pull")
def cloud_pull(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        _todo("pull")


@skills_app.command("sync")
def cloud_skills_sync(agent: str = typer.Option(None, "--agent")) -> None:
    """Sync the managed skill catalog into a harness (in development)."""
    with contract():
        _todo("skills.sync")


__all__ = ["cloud_app"]
