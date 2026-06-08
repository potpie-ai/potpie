"""Compact static setup intro logo."""

from adapters.inbound.cli.ui.logo_rotation import _intro_logo_grid, render_intro_logo
from adapters.inbound.cli.ui.potpie_logo_anim import _warming_line
from adapters.inbound.cli.ui.static_logo_loader import layout_logo_lines, load_raw_logo_lines


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
