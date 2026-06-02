"""Potpie logo terminal display (static SVG-derived ASCII, with fallbacks)."""

from __future__ import annotations

import time

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from adapters.inbound.cli.static_logo_loader import VIEWPORT_WIDTH, load_static_logo

_ACCENT = "bold #D4FF4D on #000000"
_DIM = "dim #9AE622"
_LOGO_STYLE = "bold #B8F032 on #000000"
_BLINK_DIM_STYLE = "dim #6D8A1B on #000000"

INTRO_SECONDS = 4.5

_MIN_PANEL_WIDTH = 58
_MAX_PANEL_WIDTH = 96
_ABSOLUTE_MIN_PANEL = 48

_FALLBACK_FRAMES: tuple[str, ...] = (
    " ▄███▐",
    " ███▌",
    " ▄██▐",
)


def _terminal_columns(console: Console) -> int:
    try:
        return max(20, int(console.size.width))
    except (AttributeError, TypeError, ValueError):
        return 80


def panel_width_for_console(console: Console) -> int:
    """Outer Rich panel width (borders included) that fits the terminal."""
    inner = _terminal_columns(console) - 2
    if inner >= _MIN_PANEL_WIDTH:
        return min(_MAX_PANEL_WIDTH, inner)
    if inner >= _ABSOLUTE_MIN_PANEL:
        return inner
    return max(20, inner)


def content_width_for_panel(panel_width: int) -> int:
    return max(24, panel_width - 4)


def _truncate_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len < 4:
        return text[:max_len]
    keep = max_len - 1
    left = keep // 2
    return f"{text[:left]}…{text[-(keep - left):]}"


def _styled_block(art: str) -> Text:
    out = Text()
    for i, line in enumerate(art.splitlines()):
        if i:
            out.append("\n")
        out.append(line, style=_LOGO_STYLE)
    return out


def _static_logo(viewport_width: int) -> Text | None:
    art = load_static_logo(viewport_width)
    if art is None:
        return None
    return art.text


def _logo_frame(viewport_width: int, *, blink_on: bool) -> Text:
    logo = _static_logo(viewport_width)
    if logo is None:
        return _styled_block(_FALLBACK_FRAMES[0 if blink_on else 2])
    frame = logo.copy()
    if not blink_on:
        frame.stylize(_BLINK_DIM_STYLE)
    return frame


def chomp_glyph(frame: int) -> Text:
    """Logo token for active checklist rows."""
    art = load_static_logo(VIEWPORT_WIDTH)
    on = (frame % 2) == 0
    if art is not None:
        style = "bold #B8F032 on #000000" if on else "dim #6D8A1B on #000000"
        return Text(f" {art.chomp_token} ", style=style)
    token = "██▐" if on else "██▌"
    return Text(f" {token} ", style=_LOGO_STYLE)


def play_intro(
    console: Console,
    *,
    subtitle: str = "Preparing your local context graph...",
    seconds: float = INTRO_SECONDS,
    panel_width: int | None = None,
) -> None:
    """Centered logo intro with a subtle blink animation."""
    width = panel_width if panel_width is not None else panel_width_for_console(console)
    content_w = content_width_for_panel(width)

    def render(frame_no: int) -> Panel:
        logo = _logo_frame(content_w, blink_on=(frame_no % 2 == 0))
        sub = _truncate_middle(subtitle, content_w)
        body = Group(
            Align.center(Text("potpie", style=_ACCENT)),
            Align.center(logo),
            Align.center(Text(sub, style=_DIM)),
            Align.center(Text("Potpie is warming up...", style="italic dim")),
        )
        return Panel(
            body,
            border_style="bright_green",
            padding=(0, 1),
            title="[bold]Setup[/bold]",
            width=width,
            expand=False,
        )

    with Live(render(0), console=console, refresh_per_second=4, transient=True) as live:
        tick = 0.0
        frame_no = 0
        while tick < seconds:
            live.update(render(frame_no))
            time.sleep(0.25)
            tick += 0.25
            frame_no += 1


def mini_logo_text() -> Text:
    """Static logo strip."""
    logo = _static_logo(VIEWPORT_WIDTH)
    if logo is not None:
        return logo
    return _styled_block(_FALLBACK_FRAMES[0])
