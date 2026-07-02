"""Render a compact Potpie logo for the setup intro (with optional wave animation)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rich.text import Text

from adapters.inbound.cli.ui.brand import LOGO_COLOR, LOGO_STYLE

LOGO_DIM_STYLE = f"dim {LOGO_COLOR}"
_INTRO_MAX_ROWS = 5
_INTRO_MAX_SCALE = 0.72
_WAVE_BAND = 5
_WAVE_STEP = 2
_BRAILLE_MASKS = (
    (0, 0, 0x01),
    (0, 1, 0x02),
    (0, 2, 0x04),
    (0, 3, 0x40),
    (1, 0, 0x08),
    (1, 1, 0x10),
    (1, 2, 0x20),
    (1, 3, 0x80),
)


@dataclass(frozen=True)
class _LogoCell:
    char: str
    r: int
    g: int
    b: int
    wave_pos: int


def pillow_available() -> bool:
    return importlib.util.find_spec("PIL") is not None


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def _base_logo_image():
    from PIL import Image, ImageDraw

    png = _assets_dir() / "potpie.png"
    if png.is_file():
        with Image.open(png) as img:
            return img.convert("RGBA")

    from adapters.inbound.cli.ui.static_logo_loader import load_raw_logo_lines

    lines = load_raw_logo_lines()
    if not lines:
        return None
    cell = 5
    h = len(lines)
    w = max(len(line) for line in lines)
    img = Image.new("RGBA", (w * cell, h * cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rgb = (182, 227, 67, 255)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch != " ":
                draw.rectangle(
                    (x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1),
                    fill=rgb,
                )
    return img


def _pixel_at(data: bytes, w: int, h: int, x: int, y: int) -> tuple[int, int, int, int]:
    if x < 0 or y < 0 or x >= w or y >= h:
        return (0, 0, 0, 0)
    i = (y * w + x) * 4
    return data[i], data[i + 1], data[i + 2], data[i + 3]


def _fit_size(img_w: int, img_h: int, max_cols: int, max_rows: int) -> tuple[int, int]:
    px_w = max(10, max_cols * 4)
    px_h = max(8, max_rows * 8)
    scale = min(px_w / img_w, px_h / img_h, _INTRO_MAX_SCALE)
    return max(10, int(img_w * scale)), max(10, int(img_h * scale))


def _braille_grid_from_rgba(
    px: bytes, out_w: int, out_h: int
) -> list[list[_LogoCell | None]]:
    rows: list[list[_LogoCell | None]] = []
    row_idx = 0
    for by in range(0, out_h, 4):
        row: list[_LogoCell | None] = []
        col_idx = 0
        for bx in range(0, out_w, 2):
            code = 0x2800
            rs = gs = bs = cnt = 0
            for dx, dy, bit in _BRAILLE_MASKS:
                r, g, b, a = _pixel_at(px, out_w, out_h, bx + dx, by + dy)
                if a >= 48:
                    code |= bit
                    rs += r
                    gs += g
                    bs += b
                    cnt += 1
            if cnt:
                row.append(
                    _LogoCell(
                        char=chr(code),
                        r=rs // cnt,
                        g=gs // cnt,
                        b=bs // cnt,
                        wave_pos=col_idx + row_idx,
                    )
                )
            else:
                row.append(None)
            col_idx += 1
        rows.append(row)
        row_idx += 1
    return rows


@lru_cache(maxsize=8)
def _intro_logo_grid(
    viewport_width: int, max_rows: int
) -> tuple[list[list[_LogoCell | None]], int]:
    """Cached braille grid and wave span for a viewport size."""
    if pillow_available():
        base = _base_logo_image()
        if base is not None:
            from PIL import Image

            tw, th = _fit_size(base.width, base.height, viewport_width, max_rows)
            resized = base.resize((tw, th), Image.Resampling.LANCZOS)
            grid = _braille_grid_from_rgba(resized.tobytes(), tw, th)
            span = max((c.wave_pos for row in grid for c in row if c), default=0) + 1
            return grid, span

    from adapters.inbound.cli.ui.static_logo_loader import (
        load_raw_logo_lines,
        layout_logo_lines,
    )

    lines = load_raw_logo_lines()
    if not lines:
        return [], 1
    trimmed = layout_logo_lines(
        lines, viewport_width=viewport_width, viewport_height=max_rows
    )
    grid = _ascii_grid_from_text(trimmed.plain, viewport_width)
    span = max((c.wave_pos for row in grid for c in row if c), default=0) + 1
    return grid, span


def _ascii_grid_from_text(
    plain: str, viewport_width: int
) -> list[list[_LogoCell | None]]:
    rows: list[list[_LogoCell | None]] = []
    for row_idx, line in enumerate(plain.splitlines()):
        row: list[_LogoCell | None] = []
        for col_idx, ch in enumerate(line[:viewport_width]):
            if ch.isspace():
                row.append(None)
            else:
                row.append(
                    _LogoCell(char=ch, r=182, g=227, b=67, wave_pos=col_idx + row_idx)
                )
        rows.append(row)
    return rows


def _cell_style(cell: _LogoCell, *, frame: int, span: int) -> str:
    head = (frame * _WAVE_STEP) % max(span, 1)
    dist = (cell.wave_pos - head) % span
    if dist < _WAVE_BAND:
        return LOGO_STYLE
    return LOGO_DIM_STYLE


def _render_grid_wave(
    grid: list[list[_LogoCell | None]], *, frame: int, span: int
) -> Text:
    out = Text()
    for row_idx, row in enumerate(grid):
        if row_idx:
            out.append("\n")
        for cell in row:
            if cell is None:
                out.append(" ")
            else:
                out.append(cell.char, style=_cell_style(cell, frame=frame, span=span))
    return out


def render_intro_logo(
    *,
    viewport_width: int,
    max_rows: int = _INTRO_MAX_ROWS,
    frame: int = 0,
) -> Text:
    """Small logo with a diagonal bright wave (transparent background)."""
    grid, span = _intro_logo_grid(viewport_width, max_rows)
    if grid:
        return _render_grid_wave(grid, frame=frame, span=span)
    return Text(" potpie ", style=LOGO_STYLE)


def render_static_intro_logo(
    *,
    viewport_width: int,
    max_rows: int = _INTRO_MAX_ROWS,
) -> Text:
    """Same layout as :func:`render_intro_logo` but fully bright (no wave)."""
    grid, _ = _intro_logo_grid(viewport_width, max_rows)
    if not grid:
        return Text(" potpie ", style=LOGO_STYLE)
    out = Text()
    for row_idx, row in enumerate(grid):
        if row_idx:
            out.append("\n")
        for cell in row:
            if cell is None:
                out.append(" ")
            else:
                out.append(cell.char, style=LOGO_STYLE)
    return out


# Back-compat alias (rotation removed).
render_rotating_logo = render_intro_logo

__all__ = [
    "render_intro_logo",
    "render_static_intro_logo",
    "render_rotating_logo",
    "pillow_available",
]
