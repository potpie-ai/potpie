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

from pathlib import Path
from urllib.parse import urlsplit

import typer

from potpie.cli.commands._common import contract, emit, fail, get_host_for
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
    path: str | None = typer.Option(None, "--path"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
        items = _skills().list(agent=agent, scope=effective_scope, path=path)
        emit(
            {
                "agent": agent,
                "scope": effective_scope,
                "skills": [
                    {
                        "id": s.id,
                        "version": s.version,
                        "installed": s.installed,
                        "installed_version": s.installed_version,
                        "drifted": s.drifted,
                    }
                    for s in items
                ],
            },
            human="\n".join(_skill_line(s) for s in items),
        )


@skills_app.command("install")
def skills_install(
    skill_id: str | None = typer.Argument(
        None, help="Install one skill by id; omit to install the recommended bundle."
    ),
    agent: str = typer.Option("claude", "--agent"),
    path: str | None = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
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
                verb="installed",
                agent=res.agent,
                changed=res.changed,
                support_files=res.metadata.get("support_files"),
                unavailable=res.metadata.get("unavailable"),
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
    path: str | None = typer.Option(None, "--path"),
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
        path = _resolve_path(path)
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
                verb="updated",
                agent=res.agent,
                changed=res.changed,
                support_files=res.metadata.get("support_files"),
                unavailable=res.metadata.get("unavailable"),
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
    path: str | None = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
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
            human=_format_skill_remove(
                agent=res.agent,
                removed=res.changed,
                support_files=res.metadata.get("support_files"),
                not_installed=res.metadata.get("not_installed"),
            ),
        )


@skills_app.command("status")
def skills_status(
    agent: str = typer.Option("claude", "--agent"),
    path: str | None = typer.Option(None, "--path"),
    scope: str = typer.Option("global", "--scope"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
        st = _skills().status(agent=agent, path=path, scope=effective_scope)
        # ``drifted`` is called out separately from ``outdated`` even though it
        # is a subset of it: the two have the same repair but different causes,
        # and "outdated" beside a skill sitting at the current version reads as
        # a bug in the report rather than as a damaged file.
        drifted = [s.id for s in st.outdated if s.drifted]
        emit(
            {
                "agent": st.agent,
                "scope": effective_scope,
                "installed": [s.id for s in st.installed],
                "missing": [s.id for s in st.missing],
                "outdated": [s.id for s in st.outdated],
                "drifted": drifted,
            },
            human=(
                f"agent={st.agent} installed={len(st.installed)} "
                f"missing={[s.id for s in st.missing]} outdated={[s.id for s in st.outdated]}"
                + (f" drifted={drifted}" if drifted else "")
            ),
        )


@skills_app.command("add")
def skills_add(source: str) -> None:
    with contract():
        res = _skills().add(source=_resolve_source(source))
        emit({"detail": res.detail}, human=res.detail or "added")


def _skill_line(skill) -> str:
    """One catalog row: the bundle's version, and what is actually installed.

    The installed version only earns a mention when it differs — printing
    ``v3 (installed v3)`` on eleven rows buries the one row where it is ``v2``.
    """
    mark = "✓" if skill.installed else " "
    line = f"  {mark} {skill.id} v{skill.version}"
    if skill.installed and skill.installed_version != skill.version:
        line = f"{line} (installed v{skill.installed_version})"
    if skill.drifted:
        line = f"{line} [modified — reinstall to repair]"
    return line


def _format_skill_operation(
    *,
    verb: str,
    agent: str,
    changed: tuple[str, ...],
    support_files: list[str] | None = None,
    unavailable: list[str] | None = None,
) -> str:
    if changed:
        line = f"{verb} Potpie skills for {agent}: {', '.join(changed)}"
    elif verb == "installed":
        line = f"Potpie skills for {agent} are already installed"
    else:
        line = f"Potpie skills for {agent} are already up to date"
    # Named, because these are files the command wrote that the caller did not
    # list — the harness instruction file and its slash commands. Silence here
    # is how ``skills install`` came to edit a user-authored CLAUDE.md without
    # anything in its output saying so.
    if support_files:
        line = f"{line}\n{verb} support files: {', '.join(support_files)}"
    # And the mirror image: a sweep that covered less than the catalog says so,
    # rather than letting "installed 10 skills" read as "installed everything".
    if unavailable:
        line = f"{line}\nnot carried by the {agent} bundle: {', '.join(unavailable)}"
    return line


def _format_skill_remove(
    *,
    agent: str,
    removed: tuple[str, ...],
    support_files: list[str] | None = None,
    not_installed: list[str] | None = None,
) -> str:
    """Say what was actually removed, including the files nobody named.

    ``removed: []`` on its own is the same answer this command gives after it
    has just removed the last skill, so a caller who typed an id that was never
    installed read "already removed" and moved on. The support files are named
    for the mirror-image reason ``install`` names them: they are files the
    command touched that no id in ``removed`` accounts for.
    """
    lines: list[str] = []
    if removed:
        lines.append(f"removed Potpie skills for {agent}: {', '.join(removed)}")
    if support_files:
        lines.append(f"removed support files: {', '.join(support_files)}")
    if not_installed:
        lines.append(
            f"not installed for {agent}, nothing to remove: {', '.join(not_installed)}"
        )
    if not lines:
        lines.append(f"Potpie skills for {agent} are already removed")
    return "\n".join(lines)


def _effective_scope(*, scope: str, path: str | None) -> str:
    normalized = scope.strip().lower() if scope else "global"
    if path and normalized == "global":
        return "project"
    return normalized


def _resolve_path(path: str | None) -> str | None:
    """Absolutise ``--path`` here, in the caller's process, and check it exists.

    Every one of these commands crosses an RPC to the daemon, which runs with
    whatever working directory it was launched from — often ``/`` or the
    directory of a terminal closed weeks ago. A relative path therefore resolved
    *there*: ``skills install --path .`` wrote eleven skill directories into the
    daemon's cwd and reported the install as done for the repo the user was
    standing in. A quoted ``~/project`` was worse still, since nothing expanded
    it and the daemon created a directory literally named ``~``.

    The caller's cwd is the only one that can be meant, and this process is the
    only one that knows it.

    Existence is checked for the same reason, one step later: the installer
    creates whatever it is pointed at, so a mistyped ``--path ~/porject``
    silently grew a whole skills tree in a directory nobody meant, reported the
    install as done, and left the real project untouched. A directory that is
    not there is a typo far more often than it is a request, so it is refused
    rather than materialised — ``mkdir`` is one command away when it really was
    a request.
    """
    if path is None:
        return None
    text = path.strip()
    if not text:
        return path
    resolved = Path(text).expanduser().resolve()
    if not resolved.exists():
        fail(
            code="validation_error",
            message=f"No such directory: {resolved}",
            next_action=(
                f"create it first with 'mkdir -p {resolved}', or pass the path "
                f"you meant to '--path'"
            ),
        )
    if not resolved.is_dir():
        fail(
            code="validation_error",
            message=f"--path expects a directory, but {resolved} is a file.",
            next_action="pass the directory that holds it",
        )
    return str(resolved)


def _resolve_source(source: str) -> str:
    """Absolutise a *local* ``skills add`` source, for the reason above.

    Whether the source exists is the manager's question — it is the layer that
    knows what makes a directory a skill — but it has to be asked about the
    right directory. ``skills add ./my-skill`` would otherwise be answered
    against the daemon's working directory, so a source sitting right beside the
    caller comes back as one that is not there. A URL is left exactly as typed.
    """
    text = (source or "").strip()
    if not text or urlsplit(text).scheme:
        return source
    return str(Path(text).expanduser().resolve())


__all__ = ["skills_app"]
