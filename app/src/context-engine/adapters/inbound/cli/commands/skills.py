"""Skill commands → ``HostShell.skills`` (SkillManager).

Skills are CLI-managed recipes; agents only ever see the advisory nudge in
``context_status``. These commands manage the catalog and per-harness installs.
"""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import contract, emit, get_host

skills_app = typer.Typer(help="CLI-managed agent skills.")


@skills_app.command("list")
def skills_list() -> None:
    with contract():
        items = get_host().skills.list()
        emit(
            {
                "skills": [
                    {"id": s.id, "version": s.version, "installed": s.installed}
                    for s in items
                ]
            },
            human="\n".join(
                f"  {'✓' if s.installed else ' '} {s.id} v{s.version}" for s in items
            ),
        )


@skills_app.command("install")
def skills_install(
    skill_id: str = typer.Argument(None),
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
) -> None:
    with contract():
        res = get_host().skills.install(agent=agent, skill_id=skill_id, path=path)
        emit(
            {"agent": res.agent, "changed": list(res.changed)},
            human=f"installed for {res.agent}: {', '.join(res.changed) or '(none)'}",
        )


@skills_app.command("update")
def skills_update(
    all_: bool = typer.Option(False, "--all"),
    agent: str = typer.Option("claude", "--agent"),
) -> None:
    with contract():
        res = get_host().skills.update(agent=agent, all_=all_)
        emit(
            {"agent": res.agent, "changed": list(res.changed)},
            human=f"updated for {res.agent}: {', '.join(res.changed) or '(none)'}",
        )


@skills_app.command("remove")
def skills_remove(skill_id: str, agent: str = typer.Option("claude", "--agent")) -> None:
    with contract():
        res = get_host().skills.remove(agent=agent, skill_id=skill_id)
        emit({"agent": res.agent, "removed": list(res.changed)}, human=f"removed {skill_id}")


@skills_app.command("status")
def skills_status(agent: str = typer.Option("claude", "--agent")) -> None:
    with contract():
        st = get_host().skills.status(agent=agent)
        emit(
            {
                "agent": st.agent,
                "installed": [s.id for s in st.installed],
                "missing": [s.id for s in st.missing],
                "outdated": [s.id for s in st.outdated],
            },
            human=(
                f"agent={st.agent} installed={len(st.installed)} "
                f"missing={[s.id for s in st.missing]} outdated={[s.id for s in st.outdated]}"
            ),
        )


@skills_app.command("add")
def skills_add(source: str) -> None:
    with contract():
        res = get_host().skills.add(source=source)
        emit({"detail": res.detail}, human=res.detail or "added")


__all__ = ["skills_app"]
