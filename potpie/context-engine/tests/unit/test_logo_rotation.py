"""Compact static setup intro logo."""

from potpie.context_engine.adapters.inbound.cli.ui.logo_rotation import _intro_logo_grid, render_intro_logo, render_static_intro_logo
from potpie.context_engine.adapters.inbound.cli.ui.potpie_logo_anim import (
    _celebration_sparkle,
    _center_on_logo,
    _logo_layout,
    _warming_line,
)
from potpie.context_engine.adapters.inbound.cli.ui.static_logo_loader import layout_logo_lines, load_raw_logo_lines


def test_intro_logo_wave_changes_between_frames() -> None:
    _intro_logo_grid.cache_clear()
    a = render_intro_logo(viewport_width=22, frame=0)
    b = render_intro_logo(viewport_width=22, frame=3)
    assert a.plain == b.plain
    assert str(a.spans) != str(b.spans)


def test_warming_dots_animate_in_sequence() -> None:
    a = _warming_line(0)
    b = _warming_line(1)
    assert a.plain == b.plain == "Potpie is warming up..."
    bright = "italic bold #B6E343"
    assert a.spans[1].style == bright
    assert b.spans[2].style == bright


def test_layout_logo_is_left_aligned_without_black_bg_style() -> None:
    lines = load_raw_logo_lines()
    assert lines is not None
    text = layout_logo_lines(lines, viewport_width=22, viewport_height=6)
    assert "on #000000" not in str(text.style)
    assert len(text.plain.splitlines()) <= 6


def test_center_on_logo_inset_matches_logo_bounds() -> None:
    _intro_logo_grid.cache_clear()
    logo = render_static_intro_logo(viewport_width=22)
    left, vis_w = _logo_layout(logo)
    assert vis_w > 0

    from rich.padding import Padding
    from rich.text import Text

    from potpie.context_engine.adapters.inbound.cli.ui.potpie_logo_anim import _SPLASH_LEFT_GAP

    label = Text("potpie")
    block = _center_on_logo(label, logo=logo, content_w=76)
    assert isinstance(block, Padding)
    expected = _SPLASH_LEFT_GAP + left + max(0, (vis_w - label.cell_len) // 2)
    assert block.left == expected


def test_celebration_sparkle_matches_static_width() -> None:
    from rich.text import Text

    static = Text("  ✦  ★  ✧  ·  ✦  ")
    animated = _celebration_sparkle(0)
    assert animated.plain == static.plain
    assert animated.cell_len == static.cell_len
