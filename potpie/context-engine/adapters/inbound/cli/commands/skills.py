"""Skill commands → ``HostShell.skills`` (SkillManager).

Skills are CLI-managed recipes; agents only ever see the advisory nudge in
``context_status``. These commands manage the catalog and per-harness installs.
"""

from __future__ import annotations

import typer

from adapters.inbound.cli.commands._common import contract, emit, get_host

skills_app = typer.Typer(help="CLI-managed agent skills.")


@skills_app.command("list")
def skills_list(
    agent: str = typer.Option("claude", "--agent"),
    scope: str = typer.Option("global", "--scope"),
    path: str = typer.Option(None, "--path"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        items = get_host().skills.list(
            agent=agent, scope=effective_scope, path=path
        )
        emit(
            {
                "agent": agent,
                "scope": effective_scope,
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
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        res = get_host().skills.install(
            agent=agent,
            skill_id=skill_id,
            path=path,
            scope=effective_scope,
        )
        emit(
            {
                "agent": res.agent,
                "scope": effective_scope,
                "changed": list(res.changed),
                "metadata": dict(res.metadata),
            },
            human=_format_skill_operation(
                verb="installed", agent=res.agent, changed=res.changed
            ),
        )


@skills_app.command("update")
def skills_update(
    all_: bool = typer.Option(False, "--all"),
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        res = get_host().skills.update(
            agent=agent, all_=all_, path=path, scope=effective_scope
        )
        emit(
            {
                "agent": res.agent,
                "scope": effective_scope,
                "changed": list(res.changed),
                "metadata": dict(res.metadata),
            },
            human=_format_skill_operation(
                verb="updated", agent=res.agent, changed=res.changed
            ),
        )


@skills_app.command("remove")
def skills_remove(
    skill_id: str,
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        res = get_host().skills.remove(
            agent=agent, skill_id=skill_id, path=path, scope=effective_scope
        )
        emit(
            {
                "agent": res.agent,
                "scope": effective_scope,
                "removed": list(res.changed),
                "metadata": dict(res.metadata),
            },
            human=f"removed {skill_id}",
        )


@skills_app.command("status")
def skills_status(
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        st = get_host().skills.status(agent=agent, path=path, scope=effective_scope)
        emit(
            {
                "agent": st.agent,
                "scope": effective_scope,
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


def _format_skill_operation(
    *, verb: str, agent: str, changed: tuple[str, ...]
) -> str:
    if changed:
        return f"{verb} Potpie skills for {agent}: {', '.join(changed)}"
    if verb == "installed":
        return f"Potpie skills for {agent} are already installed"
    return f"Potpie skills for {agent} are already up to date"


def _effective_scope(*, scope: str, path: str | None) -> str:
    normalized = scope.strip().lower() if scope else "global"
    if path and normalized == "global":
        return "project"
    return normalized


__all__ = ["skills_app"]
