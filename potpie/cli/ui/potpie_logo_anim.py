"""Potpie logo terminal display (compact splash with side inset, static)."""

from __future__ import annotations

import time
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.text import Text

from potpie.cli.ui.brand import LOGO_COLOR, LOGO_DIM_STYLE, LOGO_STYLE, UI_MUTED_STYLE
from potpie.cli.ui.logo_rotation import render_intro_logo, render_static_intro_logo
from potpie.cli.ui.static_logo_loader import VIEWPORT_WIDTH, load_static_logo

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
_SPLASH_LEFT_GAP = 4


def _left_splash(renderable, width: int):
    """Left-align splash content with a small inset from the border."""
    inner_w = max(1, width - _SPLASH_LEFT_GAP)
    return Padding(Align.left(renderable, width=inner_w), (0, 0, 0, _SPLASH_LEFT_GAP))


def _logo_layout(logo: Text) -> tuple[int, int]:
    """Return ``(left_offset, visual_width)`` of non-space ink inside a logo block."""
    lines = [line for line in logo.plain.splitlines() if line.strip()]
    if not lines:
        return 0, _INTRO_LOGO_COLS
    left = min(len(line) - len(line.lstrip()) for line in lines)
    right = max(len(line.rstrip()) for line in lines)
    return left, max(1, right - left)


def _center_on_logo(renderable, *, logo: Text, content_w: int):
    """Center a short label over the logo artwork (same left edge as the logo block)."""
    left_off, vis_w = _logo_layout(logo)
    label_w = renderable.cell_len
    inset = _SPLASH_LEFT_GAP + left_off + max(0, (vis_w - label_w) // 2)
    return Padding(renderable, (0, 0, 0, inset))


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
    """Left-aligned logo splash (shown briefly, then cleared)."""
    width = panel_width if panel_width is not None else panel_width_for_console(console)
    content_w = content_width_for_panel(width)
    sub = _truncate_middle(subtitle, content_w)

    def render(frame_no: int) -> Group:
        logo = _intro_logo(content_w, frame_no)
        return Group(
            Text(""),
            _center_on_logo(Text("potpie", style=_ACCENT), logo=logo, content_w=content_w),
            Text(""),
            _left_splash(logo, content_w),
            Text(""),
            _left_splash(Text(sub, style=_DIM), content_w),
            _left_splash(_warming_line(frame_no), content_w),
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
        style = _ACCENT if (frame + i) % 2 else _DIM
        out.append(glyphs[(frame + i) % len(glyphs)], style=style)
        out.append("  ")
    return out


def _finish_screen(
    *,
    headline: str,
    subline: str,
    logo: Text,
    sparkle_frame: int | None,
    content_w: int,
    agent_hint: str | None = None,
) -> Group:
    """Celebration layout; ``sparkle_frame`` animates sparkles, ``None`` keeps them fixed."""
    if sparkle_frame is None:
        top_sparkle = Text("  ✦  ★  ✧  ·  ✦  ", style=_ACCENT)
        bottom_sparkle = Text("  ✦  ★  ✧  ·  ✦  ", style=_ACCENT)
    else:
        top_sparkle = _celebration_sparkle(sparkle_frame)
        bottom_sparkle = _celebration_sparkle(sparkle_frame + 2)
    body: list[Any] = [
        Text(""),
        _center_on_logo(top_sparkle, logo=logo, content_w=content_w),
        _center_on_logo(Text("potpie", style=_ACCENT), logo=logo, content_w=content_w),
        Text(""),
        _left_splash(logo, content_w),
        Text(""),
        _left_splash(Text.from_markup(headline), content_w),
        _center_on_logo(Text(subline, style=_ACCENT), logo=logo, content_w=content_w),
        _center_on_logo(bottom_sparkle, logo=logo, content_w=content_w),
    ]
    if agent_hint:
        body.append(Text(agent_hint, style=UI_MUTED_STYLE))
    body.append(Text(""))
    return Group(*body)


def play_setup_finish(
    console: Console,
    *,
    pot_name: str,
    agent_hint: str | None = None,
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
            content_w=content_w,
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
            content_w=content_w,
            agent_hint=agent_hint,
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
