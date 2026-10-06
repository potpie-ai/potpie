"""Skill commands through the Potpie-owned skill service.

Skills are CLI-managed recipes; agents only ever see the advisory nudge in
``context_status``. These commands manage the catalog and per-harness installs.

Every skill command is filesystem-only and runs in this process: a skill install
writes files into *this* machine's harness directory (``~/.claude/skills`` and
friends), so it never contacts the daemon. ``--no-daemon`` on ``install`` is
accepted and does nothing; it stays, hidden, because installers that predate
that still pass it.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import typer

from potpie.cli.commands._common import contract, emit, fail, get_skill_service
from potpie.cli.telemetry.onboarding_events import (
    agent_skills_failure_kind,
    capture_agent_skills_install_outcome,
    elapsed_ms,
    now_ms,
)
from potpie.cli.telemetry.usage_events import capture_usage_command_succeeded

skills_app = typer.Typer(help="CLI-managed agent skills.")


@skills_app.command("list")
def skills_list(
    agent: str = typer.Option("claude", "--agent"),
    scope: str = typer.Option("global", "--scope"),
    path: str | None = typer.Option(None, "--path"),
) -> None:
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
        items = get_skill_service().list(agent=agent, scope=effective_scope, path=path)
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
                        "disabled": s.disabled,
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
    no_daemon: bool = typer.Option(
        False,
        "--no-daemon",
        hidden=True,
        help="Accepted for compatibility; installs never contact the daemon.",
    ),
) -> None:
    del no_daemon  # every install is daemon-free; see the module docstring
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
        started_ms = now_ms()
        try:
            res = get_skill_service().install(
                agent=agent,
                skill_id=skill_id,
                path=path,
                scope=effective_scope,
            )
        except (KeyboardInterrupt, EOFError):
            capture_agent_skills_install_outcome(
                agent=agent,
                entrypoint="direct_command",
                scope=effective_scope,
                outcome="cancelled",
                duration_ms=elapsed_ms(started_ms),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            capture_agent_skills_install_outcome(
                agent=agent,
                entrypoint="direct_command",
                scope=effective_scope,
                outcome="failed",
                duration_ms=elapsed_ms(started_ms),
                failure_kind=agent_skills_failure_kind(exc),
            )
            raise
        capture_agent_skills_install_outcome(
            agent=res.agent,
            entrypoint="direct_command",
            scope=effective_scope,
            outcome="installed" if res.changed else "already_installed",
            duration_ms=elapsed_ms(started_ms),
        )
        capture_usage_command_succeeded(
            command="skills install",
            result_kind="skills_result",
            item_count=len(res.changed),
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

    Mirrors ``install``/``remove``: an id names one skill, its absence means the
    set the product chooses. ``--all`` is only ever the explicit spelling of
    that default, which is why the manager refuses it *together with* an id
    instead of letting one silently discard the other.
    """
    with contract():
        effective_scope = _effective_scope(scope=scope, path=path)
        path = _resolve_path(path)
        res = get_skill_service().update(
            agent=agent,
            skill_id=skill_id,
            all_=all_,
            path=path,
            scope=effective_scope,
        )
        capture_usage_command_succeeded(
            command="skills update",
            result_kind="skills_result",
            item_count=len(res.changed),
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
        res = get_skill_service().remove(
            agent=agent,
            skill_id=skill_id,
            all_=all_,
            path=path,
            scope=effective_scope,
        )
        capture_usage_command_succeeded(
            command="skills remove",
            result_kind="skills_result",
            item_count=len(res.changed),
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
        st = get_skill_service().status(agent=agent, path=path, scope=effective_scope)
        # ``drifted`` is called out separately from ``outdated`` even though it
        # is a subset of it: the two have the same repair but different causes,
        # and "outdated" beside a skill sitting at the current version reads as
        # a bug in the report rather than as a damaged file.
        drifted = [s.id for s in st.outdated if s.drifted]
        disabled = [s.id for s in st.disabled]
        emit(
            {
                "agent": st.agent,
                "scope": effective_scope,
                "installed": [s.id for s in st.installed],
                "missing": [s.id for s in st.missing],
                "outdated": [s.id for s in st.outdated],
                "drifted": drifted,
                "disabled": disabled,
            },
            human=(
                f"agent={st.agent} installed={len(st.installed)} "
                f"missing={[s.id for s in st.missing]} "
                f"outdated={[s.id for s in st.outdated]}"
                + (f" drifted={drifted}" if drifted else "")
                + (f" disabled={disabled}" if disabled else "")
            ),
        )


@skills_app.command("add")
def skills_add(source: str) -> None:
    with contract():
        res = get_skill_service().add(source=_resolve_source(source))
        emit({"detail": res.detail}, human=res.detail or "added")


def _skill_line(skill) -> str:
    """One catalog row: the bundle's version, and what is actually installed.

    The installed version only earns a mention when it differs — printing
    ``v3 (installed v3)`` on every row buries the one row where it is ``v2``.
    """
    mark = "✓" if skill.installed else " "
    line = f"  {mark} {skill.id} v{skill.version}"
    if skill.installed and skill.installed_version != skill.version:
        line = f"{line} (installed v{skill.installed_version})"
    if skill.drifted:
        line = f"{line} [modified — preserved by update; install explicitly to replace]"
    if skill.disabled:
        line = f"{line} [disabled]"
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
    # list — the harness instruction file and its slash commands.
    if support_files:
        line = f"{line}\n{verb} support files: {', '.join(support_files)}"
    # And the mirror image: a sweep that covered less than the catalog says so,
    # rather than letting "installed N skills" read as "installed everything".
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
    has just removed the last skill, so an id that was never installed is named.
    The support files are named for the mirror-image reason ``install`` names
    them: they are files the command touched that no id in ``removed`` covers.
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
    """Absolutise ``--path`` against the caller's cwd and check it exists.

    A quoted ``~/project`` is expanded here, so nothing downstream creates a
    directory literally named ``~``. Existence is checked because the installer
    creates whatever it is pointed at: a mistyped ``--path ~/porject`` would
    grow a whole skills tree in a directory nobody meant and report the install
    as done. A directory that is not there is a typo far more often than it is
    a request, so it is refused rather than created — ``mkdir`` is one command
    away when it really was a request.
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
    """Absolutise a *local* ``skills add`` source against the caller's cwd.

    Whether the source is a skill is the manager's question, but it has to be
    asked about the directory the caller meant. A URL is left exactly as typed.
    """
    text = (source or "").strip()
    if not text or urlsplit(text).scheme:
        return source
    return str(Path(text).expanduser().resolve())


__all__ = ["skills_app"]
