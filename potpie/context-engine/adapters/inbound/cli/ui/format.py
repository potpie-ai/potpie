"""Potpie CLI visual language — shared with the setup wizard."""

from __future__ import annotations

import re
from typing import Any, Sequence

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from adapters.inbound.cli.ui.brand import (
    LOGO_COLOR,
    LOGO_DIM_STYLE,
    LOGO_STYLE,
    LOGO_SUBTLE_SEPARATOR_STYLE,
    UI_FOCUS_STYLE,
    UI_MUTED_STYLE,
)
from adapters.inbound.cli.ui.setup_wizard_ui import rich_ui_enabled

PANEL_BORDER = LOGO_COLOR
_SUBTLE_SEP_CHAR = "─"

_stdout = Console()
_stderr = Console(stderr=True)

_EMPTY_RE = re.compile(r"^\([^)]+\)$")
_ACTIVE_LABEL_RE = re.compile(r"^active:\s+(.+)$", re.IGNORECASE)
_KEY_VALUE_LINE_RE = re.compile(r"^([a-z][a-z0-9_-]*):\s+(.+)$", re.IGNORECASE)


def stdout_console() -> Console:
    return _stdout


def stderr_console() -> Console:
    return _stderr


def use_rich(*, as_json: bool = False) -> bool:
    return rich_ui_enabled(as_json=as_json)


def success_markup(message: str) -> str:
    return f"[{LOGO_COLOR}]✓[/{LOGO_COLOR}] {message}"


def step_markup(message: str) -> str:
    return f"[{LOGO_DIM_STYLE}]›[/{LOGO_DIM_STYLE}] {message}"


def muted_markup(message: str) -> str:
    return f"[{UI_MUTED_STYLE}]{escape(message)}[/{UI_MUTED_STYLE}]"


def title_markup(message: str) -> str:
    return f"[bold]{escape(message)}[/bold]"


def section_title_markup(label: str) -> str:
    return f"[bold {LOGO_COLOR}]{escape(label)}[/bold {LOGO_COLOR}]"


def error_markup(title: str, message: str) -> tuple[str, str]:
    return f"[bold red]✗ {escape(title)}[/bold red]", f"[red]{escape(message)}[/red]"


def subtle_separator(*, width: int = 48) -> Text:
    return Text(_SUBTLE_SEP_CHAR * max(24, width), style=LOGO_SUBTLE_SEPARATOR_STYLE)


def infer_line_tone(message: str) -> str:
    """Guess presentation tone from message text (success / step / muted / plain)."""
    text = message.strip()
    lower = text.lower()
    if _EMPTY_RE.match(text):
        return "muted"
    if lower.startswith(
        (
            "logged in",
            "logged out",
            "saved ",
            "authenticated",
            "connected",
            "created ",
            "active pot",
            "renamed",
            "archived",
            "added ",
            "removed ",
            "set ",
            "restarted",
            "stopped",
            "queued ",
            "mutations applied",
        )
    ):
        return "success"
    if lower.startswith(
        (
            "opening",
            "waiting",
            "copy ",
            "paste ",
            "select ",
            "enter ",
            "fetch ",
            "installing",
            "could not open",
        )
    ):
        return "step"
    if lower.startswith("error:"):
        return "plain"
    if text.endswith(":") and len(text) < 72:
        return "title"
    return "plain"


def format_key_value_line(message: str) -> str | None:
    """``label: value`` diagnostic rows — green label, white value (``doctor``)."""
    match = _KEY_VALUE_LINE_RE.match(message.strip())
    if match is None:
        return None
    key, value = match.groups()
    return (
        f"[{LOGO_COLOR}]{escape(key)}:[/{LOGO_COLOR}] "
        f"[{UI_FOCUS_STYLE}]{escape(value)}[/{UI_FOCUS_STYLE}]"
    )


def format_active_label_line(message: str) -> str | None:
    """``active: name (id)`` — green label, white value (``pot info``)."""
    match = _ACTIVE_LABEL_RE.match(message.strip())
    if match is None:
        return None
    return (
        f"[{LOGO_COLOR}]active:[/{LOGO_COLOR}] "
        f"[{UI_FOCUS_STYLE}]{escape(match.group(1))}[/{UI_FOCUS_STYLE}]"
    )


def format_line(message: str, *, tone: str | None = None) -> str:
    if tone is None:
        labeled = format_active_label_line(message)
        if labeled is not None:
            return labeled
    tone = tone or infer_line_tone(message)
    if tone == "success":
        return success_markup(message)
    if tone == "step":
        return step_markup(message)
    if tone == "muted":
        return muted_markup(message)
    if tone == "title":
        return title_markup(message)
    if tone == "warn":
        return f"[yellow]![/yellow] {escape(message)}"
    return escape(message) if tone == "plain_raw" else message


def print_line(
    message: str,
    *,
    as_json: bool = False,
    tone: str | None = None,
    markup: bool = True,
    console: Console | None = None,
) -> None:
    if as_json:
        return
    c = console or _stdout
    if not use_rich(as_json=as_json) or not markup:
        c.print(message, markup=False)
        return
    c.print(format_line(message, tone=tone))


def print_lines(
    lines: Sequence[str],
    *,
    as_json: bool = False,
    console: Console | None = None,
) -> None:
    c = console or _stdout
    if not use_rich(as_json=as_json):
        for line in lines:
            c.print(line)
        return
    for line in lines:
        if not line.strip():
            c.print()
            continue
        c.print(format_line(line))


def format_list_line(line: str) -> str:
    """Style a list row: inactive rows white; current row green with a white star."""
    stripped = line.lstrip()
    if stripped.startswith("*"):
        rest = stripped[1:].lstrip()
        return (
            f"[{UI_FOCUS_STYLE}]*[/{UI_FOCUS_STYLE}] "
            f"[{LOGO_COLOR}]{escape(rest)}[/{LOGO_COLOR}]"
        )
    if stripped.startswith("-"):
        rest = stripped[1:].lstrip()
        return (
            f"[{UI_FOCUS_STYLE}]–[/{UI_FOCUS_STYLE}] "
            f"[{UI_FOCUS_STYLE}]{escape(rest)}[/{UI_FOCUS_STYLE}]"
        )
    if line.startswith(" "):
        return f"[{UI_FOCUS_STYLE}]{escape(line)}[/{UI_FOCUS_STYLE}]"
    if _EMPTY_RE.match(stripped):
        return muted_markup(stripped)
    return format_line(line)


def _looks_like_list_row(line: str) -> bool:
    if not line.strip():
        return True
    stripped = line.lstrip()
    if stripped.startswith(("*", "-")) or _EMPTY_RE.match(stripped):
        return True
    return line[0] in (" ", "\t")


def _looks_like_key_value_line(line: str) -> bool:
    return bool(_KEY_VALUE_LINE_RE.match(line.strip()))


def print_human_block(text: str, *, console: Console | None = None) -> None:
    """Render ``emit()`` human strings with setup-aligned styling."""
    c = console or _stdout
    if not use_rich():
        c.print(text)
        return

    if "\n" not in text:
        c.print(format_line(text))
        return

    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if (
        len(non_empty) >= 2
        and not _looks_like_list_row(non_empty[0])
        and all(_looks_like_list_row(ln) for ln in non_empty[1:])
    ):
        c.print(section_title_markup(non_empty[0].rstrip(":")))
        for line in non_empty[1:]:
            c.print(format_list_line(line))
        return

    if all(_looks_like_list_row(ln) for ln in lines):
        for line in lines:
            if not line.strip():
                c.print()
            else:
                c.print(format_list_line(line))
        return

    if non_empty and all(_looks_like_key_value_line(ln) for ln in non_empty):
        for line in lines:
            if line.strip():
                c.print(format_key_value_line(line) or format_line(line))
        return

    c.print(format_line(lines[0], tone="title"))
    for line in lines[1:]:
        if line.strip():
            c.print(f"[{UI_MUTED_STYLE}]{escape(line)}[/{UI_MUTED_STYLE}]")


def key_value_panel(
    title: str,
    rows: Sequence[tuple[str, str]],
    *,
    border_style: str = PANEL_BORDER,
) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for key, value in rows:
        table.add_row(
            f"[{UI_MUTED_STYLE}]{escape(key)}[/{UI_MUTED_STYLE}]",
            escape(value),
        )
    return Panel(table, title=title, border_style=border_style)


def print_key_value_panel(
    title: str,
    rows: Sequence[tuple[str, str]],
    *,
    border_style: str = PANEL_BORDER,
    console: Console | None = None,
) -> None:
    c = console or _stdout
    c.print(key_value_panel(title, rows, border_style=border_style))


def print_structured_error(
    *,
    title: str,
    message: str,
    hint: str | None = None,
    next_action: str | None = None,
    console: Console | None = None,
) -> None:
    c = console or _stderr
    if not use_rich():
        c.print(f"error: {title}")
        if message and message != title:
            c.print(f"  {message}")
        if hint:
            c.print(f"  detail: {hint}")
        if next_action:
            c.print(f"  next: {next_action}")
        return
    head, body = error_markup(title, message if message != title else "")
    c.print(head)
    if body.strip():
        c.print(body)
    if hint:
        c.print(f"[{UI_MUTED_STYLE}]  {escape(hint)}[/{UI_MUTED_STYLE}]")
    if next_action:
        c.print(
            f"[{LOGO_DIM_STYLE}]  → {escape(next_action)}[/{LOGO_DIM_STYLE}]"
        )


__all__ = [
    "PANEL_BORDER",
    "error_markup",
    "format_active_label_line",
    "format_key_value_line",
    "format_line",
    "format_list_line",
    "infer_line_tone",
    "key_value_panel",
    "muted_markup",
    "print_human_block",
    "print_key_value_panel",
    "print_line",
    "print_lines",
    "print_structured_error",
    "section_title_markup",
    "stderr_console",
    "step_markup",
    "stdout_console",
    "subtle_separator",
    "success_markup",
    "title_markup",
    "use_rich",
]
