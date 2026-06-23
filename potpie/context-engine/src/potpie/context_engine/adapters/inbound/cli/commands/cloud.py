"""Cloud / managed-profile commands — skeleton (TODO).

Cloud is opt-in and visibly scoped (cli-flow.md). Managed routing is not built;
these commands return the structured not-implemented contract so the command
surface is complete and honest.

    TODO(stage-N): managed API routing for the shared command groups, plus
    cloud login/status/push/pull and skill sync.
"""

from __future__ import annotations

import typer

from potpie.context_engine.adapters.inbound.cli.commands._common import contract
from potpie.context_engine.domain.errors import CapabilityNotImplemented

cloud_app = typer.Typer(help="Managed profile + sync (TODO).")
skills_app = typer.Typer(help="Managed skill catalog sync (TODO).")
cloud_app.add_typer(skills_app, name="skills")


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
    """Sync the managed skill catalog into a harness (managed profile; TODO)."""
    with contract():
        _todo("skills.sync")


__all__ = ["cloud_app"]
