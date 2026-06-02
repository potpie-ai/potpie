"""POC: `potpie setup` onboarding wizard with stub provisioning (Phase 0)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import typer

from adapters.inbound.cli.credentials_store import config_dir, credentials_path
from adapters.inbound.cli.setup_wizard_ui import (
    ChecklistStep,
    SetupWizardUI,
    is_interactive_tty,
    print_json,
    rich_ui_enabled,
)

_SETUP_NEXT_COMMAND = "potpie status"

_SETUP_FILENAME = "setup.json"
_DAEMON_FILENAME = "daemon.json"
_POC_VERSION = 1
_MIN_PYTHON = (3, 10)
_MAX_PYTHON = (3, 14)


def data_dir() -> Path:
    base = os.getenv("POTPIE_DATA_HOME")
    if base:
        return Path(base).expanduser().resolve()
    return config_dir() / "data"


def setup_state_path() -> Path:
    return config_dir() / _SETUP_FILENAME


def read_setup_state() -> dict[str, Any]:
    path = setup_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_setup_state(state: dict[str, Any]) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = setup_state_path()
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _python_version_ok() -> tuple[bool, str]:
    v = sys.version_info[:3]
    if v[:2] < _MIN_PYTHON[:2]:
        return False, f"Python {v[0]}.{v[1]} < {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}"
    if (v[0], v[1]) >= _MAX_PYTHON[:2]:
        return False, f"Python {v[0]}.{v[1]} not supported (need < {_MAX_PYTHON[0]}.{_MAX_PYTHON[1]})"
    return True, f"{v[0]}.{v[1]}.{v[2]}"


def _potpie_on_path() -> tuple[str, str]:
    found = shutil.which("potpie")
    if found:
        return "ok", found
    return "warn", "not on PATH (pipx install potpie)"


def _config_writable() -> tuple[bool, str]:
    d = config_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, str(d)
    except OSError as exc:
        return False, str(exc)


def _git_repo_ok(repo: Path) -> tuple[bool, str]:
    if (repo / ".git").is_dir():
        return True, str(repo.resolve())
    return False, "not a git repo"


def _step_dict(step: ChecklistStep) -> dict[str, Any]:
    return {
        "id": step.step_id,
        "name": step.label,
        "status": step.status,
        "detail": step.detail,
        "duration_ms": step.duration_ms,
    }


def _stub_ensure_config() -> str:
    config_dir().mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    return str(config_dir())


def _stub_local_auth() -> str:
    cred_path = credentials_path()
    if not cred_path.is_file():
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        cred_path.write_text(
            json.dumps({"local_profile": True, "local_token": uuid.uuid4().hex}, indent=2) + "\n",
            encoding="utf-8",
        )
    return "local auth ready"


def _stub_daemon() -> str:
    payload = {
        "stub": True,
        "pid": os.getpid(),
        "socket": str(config_dir() / "daemon.sock"),
        "health": "ok",
    }
    path = config_dir() / _DAEMON_FILENAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return "daemon ready"


def _stub_state_db() -> str:
    db_path = data_dir() / "state" / "potpie.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.is_file():
        db_path.write_bytes(b"")
    return "state DB ready"


def _stub_graph() -> str:
    graph = data_dir() / "graph"
    vector = data_dir() / "vector"
    graph.mkdir(parents=True, exist_ok=True)
    vector.mkdir(parents=True, exist_ok=True)
    (graph / ".gitkeep").touch(exist_ok=True)
    (vector / ".gitkeep").touch(exist_ok=True)
    return "context graph ready"


def _run_provision_step(
    wizard: SetupWizardUI,
    step_id: str,
    fn: Callable[[], str],
    *,
    min_display_s: float = 0.0,
) -> bool:
    step = wizard.get(step_id)
    try:
        with wizard.run_step(step_id):
            t0 = time.perf_counter()
            detail = fn()
            if min_display_s > 0:
                remain = min_display_s - (time.perf_counter() - t0)
                if remain > 0:
                    time.sleep(remain)
        step.status = "done"
        step.detail = detail
        return True
    except OSError as exc:
        step.status = "failed"
        step.detail = str(exc)
        return False


def _silent_preflight(repo: Path) -> tuple[bool, str | None]:
    """Run technical checks before the visible checklist (no UI)."""
    ok, detail = _python_version_ok()
    if not ok:
        return False, f"Python: {detail}"
    ok, detail = _config_writable()
    if not ok:
        return False, f"Config dir: {detail}"
    return True, None


def run_setup_poc(
    *,
    repo: Path,
    pot_name: str,
    agent: str,
    scan: bool,
    yes: bool,
    as_json: bool,
) -> None:
    use_rich = rich_ui_enabled(as_json=as_json)
    wizard = SetupWizardUI(use_rich=use_rich)

    existing = read_setup_state()
    already_complete = existing.get("status") == "complete" and existing.get("poc_version") == _POC_VERSION

    # User-facing checklist (matches product copy)
    wizard.add_step("repo", "Repo detected", done_label="Repo detected")
    wizard.add_step("agent", f"Agent selected: {agent}", done_label=f"Agent selected: {agent}")
    scan_label = "Scan enabled" if scan else "Scan disabled"
    wizard.add_step("scan_plan", scan_label, done_label=scan_label)
    wizard.add_step("files", "Creating config files…", done_label="Config files ready")
    wizard.add_step("auth", "Initializing local auth…", done_label="Local auth ready")
    wizard.add_step(
        "daemon",
        "Starting daemon…",
        chomp=True,
        done_label="Daemon ready",
    )
    wizard.add_step(
        "state_db",
        "Preparing state DB…",
        chomp=True,
        done_label="State DB ready",
    )
    wizard.add_step(
        "graph",
        "Building context graph…",
        chomp=True,
        done_label="Graph ready",
    )
    wizard.add_step(
        "pot",
        "Creating default pot…",
        done_label="Default pot created",
    )
    if scan:
        wizard.add_step("scan", "Scanning repository…", done_label="Repository scanned")
    wizard.add_step("skills", "Installing agent skills (stub)…", done_label="Skills noted")

    if not as_json:
        wizard.run_intro(repo=repo, agent=agent, scan=scan)

    if already_complete and yes:
        for s in wizard.steps:
            s.status = "skipped"
            s.detail = "already set up"
        _emit_result(
            wizard=wizard,
            as_json=as_json,
            ok=True,
            already_setup=True,
            pot_name=str(existing.get("active_pot") or pot_name),
        )
        return

    ok, err = _silent_preflight(repo)
    if not ok:
        if as_json:
            print_json(
                {
                    "ok": False,
                    "code": "preflight_failed",
                    "message": err,
                    "recommended_next_action": "potpie setup --repo .",
                }
            )
        else:
            typer.secho(f"Preflight failed: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if not yes and is_interactive_tty() and use_rich and not already_complete:
        typer.confirm(
            f"Set up Potpie for {repo.resolve()} (pot={pot_name}, agent={agent})?",
            default=True,
            abort=True,
        )

    chomp_min = 0.75

    with wizard.live():
        repo_ok, repo_detail = _git_repo_ok(repo)
        wizard.get("repo").status = "warn" if not repo_ok else "done"
        wizard.get("repo").detail = repo_detail
        wizard.get("agent").status = "done"
        wizard.get("scan_plan").status = "done"
        wizard.refresh()
        time.sleep(0.35)

        if already_complete:
            for s in wizard.steps:
                if s.step_id in ("repo", "agent", "scan_plan"):
                    continue
                s.status = "skipped"
                s.detail = "already set up"
        else:
            if not _run_provision_step(wizard, "files", _stub_ensure_config):
                wizard.print_failed(step_id="files")
                _emit_result(
                    wizard=wizard,
                    as_json=as_json,
                    ok=False,
                    already_setup=False,
                    pot_name=pot_name,
                )
                return
            if not _run_provision_step(wizard, "auth", _stub_local_auth):
                wizard.print_failed(step_id="auth")
                _emit_result(
                    wizard=wizard,
                    as_json=as_json,
                    ok=False,
                    already_setup=False,
                    pot_name=pot_name,
                )
                return
            if not _run_provision_step(
                wizard, "daemon", _stub_daemon, min_display_s=chomp_min
            ):
                wizard.print_failed(step_id="daemon")
                _emit_result(
                    wizard=wizard,
                    as_json=as_json,
                    ok=False,
                    already_setup=False,
                    pot_name=pot_name,
                )
                return
            if not _run_provision_step(
                wizard, "state_db", _stub_state_db, min_display_s=chomp_min
            ):
                wizard.print_failed(step_id="state_db")
                _emit_result(
                    wizard=wizard,
                    as_json=as_json,
                    ok=False,
                    already_setup=False,
                    pot_name=pot_name,
                )
                return
            if not _run_provision_step(
                wizard, "graph", _stub_graph, min_display_s=chomp_min
            ):
                wizard.print_failed(step_id="graph")
                _emit_result(
                    wizard=wizard,
                    as_json=as_json,
                    ok=False,
                    already_setup=False,
                    pot_name=pot_name,
                )
                return

            with wizard.run_step("pot"):
                time.sleep(0.3)
                pot_step = wizard.get("pot")
                pot_step.status = "done"
                pot_step.detail = pot_name

            if scan:
                with wizard.run_step("scan"):
                    time.sleep(0.6)
                    wizard.get("scan").status = "done"
                    wizard.get("scan").detail = "stub"

            with wizard.run_step("skills"):
                wizard.get("skills").status = "done"
                wizard.get("skills").detail = "potpie init-agent"

    if not already_complete:
        write_setup_state(
            {
                "poc_version": _POC_VERSION,
                "status": "complete",
                "active_pot": pot_name,
                "repo": str(repo.resolve()),
                "agent": agent,
                "backend": "embedded",
                "scan_enabled": scan,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    # Offer GitHub login after successful setup (interactive mode only)
    if not as_json and is_interactive_tty():
        _prompt_github_login()

    _emit_result(
        wizard=wizard,
        as_json=as_json,
        ok=True,
        already_setup=already_complete,
        pot_name=pot_name,
    )


def _prompt_github_login() -> None:
    """After setup, offer to connect GitHub via device flow."""
    from adapters.inbound.cli.auth.github_commands import github_login_impl

    try:
        confirmed = typer.confirm(
            "\nWould you like to log in to GitHub now?",
            default=True,
        )
    except typer.Abort:
        confirmed = False

    if confirmed:
        github_login_impl()


def _emit_preflight_fail(wizard: SetupWizardUI, *, as_json: bool) -> None:
    payload = {
        "ok": False,
        "code": "preflight_failed",
        "message": "One or more preflight checks failed",
        "steps": [_step_dict(s) for s in wizard.steps],
        "recommended_next_action": "Fix the failed checks and re-run: potpie setup",
    }
    if as_json:
        print_json(payload)
    else:
        if not wizard.use_rich:
            wizard.print_plain_steps()
        typer.secho("Preflight failed.", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _emit_result(
    *,
    wizard: SetupWizardUI,
    as_json: bool,
    ok: bool,
    already_setup: bool,
    pot_name: str,
) -> None:
    if as_json:
        print_json(
            {
                "ok": ok,
                "already_setup": already_setup,
                "config_path": str(setup_state_path()),
                "data_home": str(data_dir()),
                "active_pot": pot_name,
                "steps": [_step_dict(s) for s in wizard.steps],
                "next_commands": [_SETUP_NEXT_COMMAND],
            }
        )
        if not ok:
            raise typer.Exit(code=1)
        return

    if not ok:
        raise typer.Exit(code=1)

    if not wizard.use_rich:
        wizard.print_plain_steps()

    wizard.print_complete_summary(
        setup_path=str(setup_state_path()),
        data_path=str(data_dir()),
        pot_name=pot_name,
        already_setup=already_setup,
    )
