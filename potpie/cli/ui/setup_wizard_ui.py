"""Unified Rich setup wizard: intro, live checklist, final summary."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.markup import escape
from rich.text import Text

from potpie.cli.ui.brand import (
    LOGO_COLOR,
    LOGO_DIM_STYLE,
    LOGO_SEPARATOR_STYLE,
    LOGO_STYLE,
    LOGO_SUBTLE_SEPARATOR_STYLE,
)
from potpie.cli.ui.potpie_logo_anim import (
    content_width_for_panel,
    panel_width_for_console,
    play_intro,
)

StepStatus = Literal["pending", "running", "done", "skipped", "failed", "warn"]

_SEPARATOR_CHAR = "━"
_SUBTLE_SEPARATOR_CHAR = "─"


def _green_separator(width: int) -> Text:
    """Thick green separator used in place of boxed panels."""
    return Text(_SEPARATOR_CHAR * max(12, width), style=LOGO_SEPARATOR_STYLE)


def _subtle_green_separator(width: int) -> Text:
    """Slim dim-green separator (matches post-setup section breaks)."""
    return Text(
        _SUBTLE_SEPARATOR_CHAR * max(24, width),
        style=LOGO_SUBTLE_SEPARATOR_STYLE,
    )


_ICON: dict[StepStatus, tuple[str, str]] = {
    "pending": ("•", LOGO_DIM_STYLE),
    "running": ("⠋", LOGO_STYLE),
    "done": ("✓", LOGO_STYLE),
    "failed": ("✗", "red"),
    "skipped": ("–", LOGO_DIM_STYLE),
    "warn": ("!", "yellow"),
}
_RUNNING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _status_icon(status: StepStatus, frame: int) -> Text:
    glyph, style = _ICON.get(status, ("•", "dim"))
    if status == "running":
        glyph = _RUNNING_FRAMES[frame % len(_RUNNING_FRAMES)]
    return Text(f"[{glyph}]", style=style)


@dataclass
class ChecklistStep:
    step_id: str
    label: str
    status: StepStatus = "pending"
    detail: str = ""
    chomp: bool = False
    done_label: str = ""
    duration_ms: int | None = None

    def display_label(self) -> str:
        if self.status == "done" and self.done_label:
            return self.done_label
        return self.label


def stderr_console() -> Console:
    force_terminal = _env_flag("POTPIE_FORCE_UI")
    return Console(
        stderr=_setup_console_uses_stderr(),
        force_terminal=True if force_terminal else None,
    )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _setup_console_uses_stderr() -> bool:
    """Prefer stderr, but use stdout when it is the only terminal stream."""
    if sys.stderr.isatty():
        return True
    if sys.stdout.isatty():
        return False
    return True


def _truncate_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len < 4:
        return text[:max_len]
    keep = max_len - 1
    left = keep // 2
    return f"{text[:left]}…{text[-(keep - left) :]}"


def rich_ui_enabled(*, as_json: bool) -> bool:
    """Rich styling when stderr/stdout is a TTY (Cursor often has non-TTY stdin)."""
    if as_json:
        return False
    if _env_flag("POTPIE_PLAIN"):
        return False
    if _env_flag("POTPIE_FORCE_UI"):
        return True
    if os.getenv("TERM", "").strip().lower() == "dumb":
        return False
    return sys.stderr.isatty() or sys.stdout.isatty()


def live_ui_enabled(*, as_json: bool) -> bool:
    """True when setup may repaint a live terminal region."""
    if _env_flag("POTPIE_NO_LIVE"):
        return False
    return rich_ui_enabled(as_json=as_json)


def is_interactive_tty() -> bool:
    """Whether to show confirmation prompts."""
    return (sys.stderr.isatty() or sys.stdout.isatty()) and sys.stdin.isatty()


def print_json(payload: dict[str, Any]) -> None:
    import json

    print(json.dumps(payload, indent=2))


class _LiveSetupRenderable:
    def __init__(self, wizard: SetupWizardUI) -> None:
        self._wizard = wizard

    def __rich_console__(self, console: Console, _options: Any) -> Iterator[Group]:
        yield self._wizard._render(console)


@dataclass
class SetupWizardUI:
    """Interactive setup checklist; no-op when ``use_rich`` is False."""

    use_rich: bool
    steps: list[ChecklistStep] = field(default_factory=list)
    _live: Live | None = field(default=None, repr=False)
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    def add_step(
        self,
        step_id: str,
        label: str,
        *,
        chomp: bool = False,
        done_label: str = "",
    ) -> ChecklistStep:
        step = ChecklistStep(
            step_id=step_id,
            label=label,
            chomp=chomp,
            done_label=done_label,
        )
        self.steps.append(step)
        return step

    def get(self, step_id: str) -> ChecklistStep:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)

    def run_intro(
        self,
        *,
        repo: Path,
        agent: str,
        scan: bool,
    ) -> None:
        if not self.use_rich:
            print("Potpie Setup")
            print("Preparing your local context graph...")
            return
        console = stderr_console()
        width = panel_width_for_console(console)
        extras = f"agent {agent}"
        if scan:
            extras += " · scan on"
        play_intro(
            console,
            subtitle=f"Repo {repo.resolve()} · {extras}",
            seconds=2.6,
            panel_width=width,
        )
        console.print()
        console.print("[bold]Potpie Setup[/bold]")
        console.print("[dim]Setting up local onboarding for this repo.[/dim]\n")

    def _render_table(self, *, content_width: int) -> Table:
        table = Table.grid(padding=(0, 1))
        label_width = max(16, content_width - 4)
        table.add_column(width=3, no_wrap=True)
        table.add_column(width=label_width, no_wrap=True, overflow="ellipsis")
        frame = int((time.monotonic() - self._started_at) / 0.12)
        for step in self.steps:
            icon = _status_icon(step.status, frame)
            text = step.display_label()
            line = Text(text, style="bold" if step.status == "running" else "")
            if step.detail:
                detail_width = max(12, label_width - len(text) - 1)
                detail = _truncate_middle(step.detail, detail_width)
                line.append(" ")
                line.append(detail, style="dim")
            table.add_row(icon, line)
        return table

    def _render(self, console: Console | None = None) -> Group:
        console = console or stderr_console()
        width = panel_width_for_console(console)
        return Group(self._render_table(content_width=content_width_for_panel(width)))

    def refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def start_step(self, step_id: str) -> None:
        """Mark a checklist row as running and refresh the live view."""
        step = self.get(step_id)
        step.status = "running"
        self.refresh()

    def complete_step(
        self,
        step_id: str,
        *,
        status: StepStatus,
        detail: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Apply a final status to a checklist row and refresh the live view."""
        step = self.get(step_id)
        step.status = status
        step.detail = detail or ""
        step.duration_ms = duration_ms
        self.refresh()

    @contextmanager
    def live(self) -> Iterator[SetupWizardUI]:
        if not self.use_rich:
            yield self
            return
        console = stderr_console()
        with Live(
            _LiveSetupRenderable(self),
            console=console,
            refresh_per_second=10,
            transient=False,
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    @contextmanager
    def run_step(self, step_id: str) -> Iterator[None]:
        """Mark step running, yield for work, caller sets done/fail."""
        self.start_step(step_id)
        step = self.get(step_id)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            step.duration_ms = int((time.perf_counter() - t0) * 1000)
            self.refresh()

    def print_plain_steps(self) -> None:
        for step in self.steps:
            if step.status == "pending":
                continue
            mark = {
                "done": "✓",
                "failed": "✗",
                "warn": "!",
                "running": "…",
                "pending": "•",
                "skipped": "–",
            }.get(step.status, "?")
            line = f"{mark} {step.display_label()}"
            if step.detail:
                line += f" — {step.detail}"
            print(line)

    def print_final_checklist(self) -> None:
        """Print checklist when not using Live (Live already leaves the final frame)."""
        if not self.use_rich or self._live is not None:
            return
        console = stderr_console()
        console.print(self._render(console))

    def print_complete_summary(
        self,
        *,
        setup_path: str,
        data_path: str,
        pot_name: str | None = None,
        already_setup: bool,
    ) -> None:
        if not self.use_rich:
            header = "Already set up." if already_setup else "Potpie setup complete."
            print(f"\n{header}")
            self.print_plain_steps()
            print("\nNext: potpie status")
            return

        console = stderr_console()
        width = panel_width_for_console(console)
        cw = content_width_for_panel(width)
        console.print()
        title = (
            f"[bold {LOGO_COLOR}]Potpie is ready![/bold {LOGO_COLOR}]"
            if not already_setup
            else "[dim]Already set up — no changes needed.[/dim]"
        )
        console.print(title)
        lines: list[str] = [
            f"[{LOGO_COLOR}]✓[/{LOGO_COLOR}] Config [cyan]{escape(_truncate_middle(setup_path, cw - 10))}[/cyan]",
            f"[{LOGO_COLOR}]✓[/{LOGO_COLOR}] Data [cyan]{escape(_truncate_middle(data_path, cw - 8))}[/cyan]",
        ]
        if pot_name:
            lines.append(
                f"[{LOGO_COLOR}]✓[/{LOGO_COLOR}] Pot [bold]{escape(pot_name)}[/bold]"
            )
        lines.extend(["", "Next: [bold]connect tools & agents[/bold]"])
        console.print("\n".join(lines))
        console.print(_subtle_green_separator(width))

    def print_failed(self, *, step_id: str, verbose_hint: bool = True) -> None:
        step = self.get(step_id)
        typer.secho(f"\n✗ {step.label} failed", fg=typer.colors.RED, err=True)
        if step.detail:
            typer.secho(f"  {step.detail}", fg=typer.colors.RED, err=True)
        if verbose_hint:
            typer.secho(
                "\nRun with verbose logs:\n  potpie -v setup --repo .",
                fg=typer.colors.YELLOW,
                err=True,
            )
