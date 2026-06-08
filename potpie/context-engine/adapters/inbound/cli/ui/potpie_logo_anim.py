"""Potpie logo terminal display (compact, centered, static)."""

from __future__ import annotations

import time

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from adapters.inbound.cli.ui.brand import LOGO_COLOR, LOGO_DIM_STYLE, LOGO_STYLE
from adapters.inbound.cli.ui.logo_rotation import render_intro_logo, render_static_intro_logo
from adapters.inbound.cli.ui.static_logo_loader import VIEWPORT_WIDTH, load_static_logo

_ACCENT = LOGO_STYLE
_DIM = LOGO_DIM_STYLE
INTRO_SECONDS = 4.5
_INTRO_LOGO_COLS = 22
_WARMING_BASE = "Potpie is warming up"
_WARMING_BRIGHT = f"italic bold {LOGO_COLOR}"
_WARMING_DIM = "italic dim"
_INTRO_FPS = 8

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
    return f"{text[:left]}…{text[-(keep - left) :]}"


def _styled_block(art: str) -> Text:
    out = Text()
    for i, line in enumerate(art.splitlines()):
        if i:
            out.append("\n")
        out.append(line, style=LOGO_STYLE)
    return out


def _warming_line(frame: int) -> Text:
    """Animate ellipsis: one dot bright at a time, the others dim."""
    phase = frame % 3
    out = Text()
    out.append(_WARMING_BASE, style=_WARMING_DIM)
    for i in range(3):
        style = _WARMING_BRIGHT if i == phase else _WARMING_DIM
        out.append(".", style=style)
    return out


def _intro_logo(viewport_width: int, frame: int) -> Text:
    logo_w = min(_INTRO_LOGO_COLS, viewport_width)
    try:
        return render_intro_logo(viewport_width=logo_w, frame=frame)
    except Exception:
        art = load_static_logo(logo_w)
        if art is not None:
            return art.text
        return _styled_block(_FALLBACK_FRAMES[frame % len(_FALLBACK_FRAMES)])


def chomp_glyph(frame: int) -> Text:
    """Logo token for active checklist rows."""
    art = load_static_logo(VIEWPORT_WIDTH)
    token = art.chomp_token if art is not None else "@@"
    return Text(f" [{token}] ", style=LOGO_STYLE)


def play_intro(
    console: Console,
    *,
    subtitle: str = "Preparing your local context graph...",
    seconds: float = INTRO_SECONDS,
    panel_width: int | None = None,
) -> None:
    """Centered static logo splash (shown briefly, then cleared)."""
    width = panel_width if panel_width is not None else panel_width_for_console(console)
    content_w = content_width_for_panel(width)
    sub = _truncate_middle(subtitle, content_w)

    def render(frame_no: int) -> Group:
        return Group(
            Text(""),
            Align.center(Text("potpie", style=_ACCENT)),
            Text(""),
            Align.center(_intro_logo(content_w, frame_no)),
            Text(""),
            Align.center(Text(sub, style=_DIM)),
            Align.center(_warming_line(frame_no)),
            Text(""),
        )

    interval = 1.0 / _INTRO_FPS
    with Live(render(0), console=console, refresh_per_second=_INTRO_FPS, transient=True) as live:
        tick = 0.0
        frame_no = 0
        while tick < seconds:
            live.update(render(frame_no))
            time.sleep(interval)
            tick += interval
            frame_no += 1


def mini_logo_text() -> Text:
    """Static logo strip."""
    art = load_static_logo(min(VIEWPORT_WIDTH, 24))
    if art is not None:
        return art.text
    return _styled_block(_FALLBACK_FRAMES[0])


_FINISH_SECONDS = 2.8
_FINISH_FPS = 8


def _celebration_sparkle(frame: int) -> Text:
    glyphs = ("✦", "★", "✧", "·")
    out = Text("  ")
    for i in range(5):
        out.append(glyphs[(frame + i) % len(glyphs)], style=_ACCENT if (frame + i) % 2 else _DIM)
        out.append(" ")
    return out


def _finish_screen(
    *,
    headline: str,
    subline: str,
    logo: Text,
    sparkle_frame: int | None,
) -> Group:
    """Celebration layout; ``sparkle_frame`` animates sparkles, ``None`` keeps them fixed."""
    if sparkle_frame is None:
        top_sparkle = Text("  ✦  ★  ✧  ·  ✦  ", style=_ACCENT)
        bottom_sparkle = Text("  ✦  ★  ✧  ·  ✦  ", style=_ACCENT)
    else:
        top_sparkle = _celebration_sparkle(sparkle_frame)
        bottom_sparkle = _celebration_sparkle(sparkle_frame + 2)
    return Group(
        Text(""),
        Align.center(top_sparkle),
        Align.center(Text("potpie", style=_ACCENT)),
        Text(""),
        Align.center(logo),
        Text(""),
        Align.center(Text.from_markup(headline)),
        Align.center(Text(subline, style=_ACCENT)),
        Align.center(bottom_sparkle),
        Text(""),
    )


def play_setup_finish(
    console: Console,
    *,
    pot_name: str,
    seconds: float = _FINISH_SECONDS,
    panel_width: int | None = None,
) -> None:
    """Animate in place, then hold the same layout as a static screen (no clear)."""
    width = panel_width if panel_width is not None else panel_width_for_console(console)
    content_w = content_width_for_panel(width)
    logo_w = min(_INTRO_LOGO_COLS, content_w)
    headline = f"Your pot [bold]{pot_name}[/bold] is ready."
    subline = "Start ingestion."

    def render_animated(frame_no: int) -> Group:
        try:
            logo = render_intro_logo(viewport_width=logo_w, frame=frame_no)
        except Exception:
            logo = _intro_logo(content_w, frame_no)
        return _finish_screen(
            headline=headline,
            subline=subline,
            logo=logo,
            sparkle_frame=frame_no,
        )

    def render_static() -> Group:
        try:
            logo = render_static_intro_logo(viewport_width=logo_w)
        except Exception:
            logo = _intro_logo(content_w, 0)
        return _finish_screen(
            headline=headline,
            subline=subline,
            logo=logo,
            sparkle_frame=None,
        )

    interval = 1.0 / _FINISH_FPS
    with Live(
        render_animated(0),
        console=console,
        refresh_per_second=_FINISH_FPS,
        transient=False,
    ) as live:
        tick = 0.0
        frame_no = 0
        while tick < seconds:
            live.update(render_animated(frame_no))
            time.sleep(interval)
            tick += interval
            frame_no += 1
        live.update(render_static(), refresh=True)
