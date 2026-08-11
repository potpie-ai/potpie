"""Skill commands → ``HostShell.skills`` (SkillManager).

Skills are CLI-managed recipes; agents only ever see the advisory nudge in
``context_status``. These commands manage the catalog and per-harness installs.

They are pinned to the local host. A skill install writes files into *this*
machine's harness directory (``~/.claude/skills`` and friends) — routed to a
managed host it would install onto the server's filesystem, where no harness of
yours will ever read them, and report success. Which graph you are pointed at
has no bearing on where your agent reads its skills from.
"""

from __future__ import annotations

import typer

from potpie.cli.commands._common import contract, emit, get_host_for
from potpie.cli.telemetry.onboarding_events import (
    capture_project_binding_event,
    elapsed_ms,
    now_ms,
    sanitized_failure_kind,
)

skills_app = typer.Typer(help="CLI-managed agent skills.")


def _skills():
    """The local host's skill manager — never the active host; see the module
    docstring."""
    from potpie.cli import hosts

    return get_host_for(hosts.LOCAL).skills


@skills_app.command("list")
def skills_list(
    agent: str = typer.Option("claude", "--agent"),
    scope: str = typer.Option("global", "--scope"),
    path: str = typer.Option(None, "--path"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        items = _skills().list(agent=agent, scope=effective_scope, path=path)
        emit(
            {
                "agent": agent,
                "scope": effective_scope,
                "skills": [
                    {"id": s.id, "version": s.version, "installed": s.installed}
                    for s in items
                ],
            },
            human="\n".join(
                f"  {'✓' if s.installed else ' '} {s.id} v{s.version}" for s in items
            ),
        )


@skills_app.command("install")
def skills_install(
    skill_id: str | None = typer.Argument(
        None, help="Install one skill by id; omit to install the recommended bundle."
    ),
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        started_ms = now_ms()
        capture_project_binding_event(
            "cli_onboarding_agent_skills_install_started",
            entrypoint="direct_command",
            properties={"agent": agent, "scope": effective_scope},
        )
        try:
            res = _skills().install(
                agent=agent,
                skill_id=skill_id,
                path=path,
                scope=effective_scope,
            )
        except Exception as exc:  # noqa: BLE001
            capture_project_binding_event(
                "cli_onboarding_agent_skills_install_failed",
                entrypoint="direct_command",
                properties={
                    "agent": agent,
                    "scope": effective_scope,
                    "failure_kind": sanitized_failure_kind(exc),
                    "duration_ms": elapsed_ms(started_ms),
                },
            )
            raise
        capture_project_binding_event(
            "cli_onboarding_agent_skills_install_completed",
            entrypoint="direct_command",
            properties={
                "agent": res.agent,
                "scope": effective_scope,
                "changed_count": len(res.changed),
                "duration_ms": elapsed_ms(started_ms),
            },
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
    skill_id: str | None = typer.Argument(
        None, help="Update one skill by id; omit to update everything installed."
    ),
    all_: bool = typer.Option(
        False,
        "--all",
        help="Update every installed Potpie skill for the selected agent and scope.",
    ),
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    """Update installed skills — one named id, or everything installed.

    The id is what makes the manager's refusal reachable. Without it click
    answered ``potpie skills update <typo>`` with "Got unexpected extra
    argument" — a statement about the command line rather than about the skill —
    while the branch that knows the id is unknown sat one call below, exercised
    only by tests that drove the manager directly. A behaviour no user can reach
    is not a behaviour the product has.

    Mirrors ``install``/``remove``: an id names one skill, its absence means the
    set the product chooses. ``--all`` is only ever the explicit spelling of
    that default, which is why the manager refuses it *together with* an id
    instead of letting one silently discard the other.
    """
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        res = _skills().update(
            agent=agent,
            skill_id=skill_id,
            all_=all_,
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
                verb="updated", agent=res.agent, changed=res.changed
            ),
        )


@skills_app.command("remove")
def skills_remove(
    skill_id: str | None = typer.Argument(None),
    all_: bool = typer.Option(
        False,
        "--all",
        help="Remove every installed Potpie skill for the selected agent and scope.",
    ),
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        res = _skills().remove(
            agent=agent,
            skill_id=skill_id,
            all_=all_,
            path=path,
            scope=effective_scope,
        )
        emit(
            {
                "agent": res.agent,
                "scope": effective_scope,
                "removed": list(res.changed),
                "metadata": dict(res.metadata),
            },
            human=_format_skill_remove(agent=res.agent, removed=res.changed),
        )


@skills_app.command("status")
def skills_status(
    agent: str = typer.Option("claude", "--agent"),
    path: str = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        st = _skills().status(agent=agent, path=path, scope=effective_scope)
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
        res = _skills().add(source=source)
        emit({"detail": res.detail}, human=res.detail or "added")


def _format_skill_operation(*, verb: str, agent: str, changed: tuple[str, ...]) -> str:
    if changed:
        return f"{verb} Potpie skills for {agent}: {', '.join(changed)}"
    if verb == "installed":
        return f"Potpie skills for {agent} are already installed"
    return f"Potpie skills for {agent} are already up to date"


def _format_skill_remove(*, agent: str, removed: tuple[str, ...]) -> str:
    if removed:
        return f"removed Potpie skills for {agent}: {', '.join(removed)}"
    return f"Potpie skills for {agent} are already removed"


def _effective_scope(*, scope: str, path: str | None) -> str:
    normalized = scope.strip().lower() if scope else "global"
    if path and normalized == "global":
        return "project"
    return normalized


__all__ = ["skills_app"]
