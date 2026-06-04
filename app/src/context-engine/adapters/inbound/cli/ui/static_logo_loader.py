"""Load static Potpie logo from SVG-derived ASCII (``potpie-logo-static.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rich.text import Text

VIEWPORT_WIDTH = 96
VIEWPORT_HEIGHT = 20
LOGO_COLOR = "#B6E343"
LOGO_STYLE = f"bold {LOGO_COLOR} on #000000"
BLANK_STYLE = "on #000000"


@dataclass(frozen=True)
class StaticLogoArt:
    width: int
    height: int
    text: Text
    chomp_token: str


def resolve_static_logo_path() -> Path | None:
    here = Path(__file__).resolve().parent
    path = here / "assets" / "potpie-logo-static.json"
    return path if path.is_file() else None


def _chomp_token_from_lines(lines: list[str]) -> str:
    preferred = "@#%*+="
    blob = "\n".join(lines)
    for ch in preferred:
        if ch in blob:
            return ch * 2
    for line in lines:
        for ch in line:
            if ch not in " .":
                return ch * 2
    return "@@"


def _center_lines_in_viewport(
    lines: list[str],
    *,
    viewport_width: int,
    viewport_height: int,
    style: str,
) -> Text:
    if not lines:
        out = Text()
        for row in range(viewport_height):
            if row:
                out.append("\n")
            out.append(" " * viewport_width, style=BLANK_STYLE)
        return out

    art_h = len(lines)
    art_w = max(len(line) for line in lines)

    if art_h > viewport_height:
        trim = art_h - viewport_height
        start = trim // 2
        lines = lines[start : start + viewport_height]
        art_h = len(lines)

    pad_top = max(0, (viewport_height - art_h) // 2)
    pad_left = max(0, (viewport_width - art_w) // 2)

    out = Text()
    for row in range(viewport_height):
        if row:
            out.append("\n")
        if row < pad_top or row >= pad_top + art_h:
            out.append(" " * viewport_width, style=BLANK_STYLE)
            continue
        src = lines[row - pad_top]
        prefix = " " * pad_left + src
        suffix_len = viewport_width - len(prefix)
        if suffix_len > 0:
            prefix += " " * suffix_len
        out.append(prefix[:viewport_width], style=style)
    return out


@lru_cache(maxsize=16)
def load_static_logo(viewport_width: int = VIEWPORT_WIDTH) -> StaticLogoArt | None:
    path = resolve_static_logo_path()
    if path is None:
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        lines = [str(line) for line in (doc.get("lines") or []) if line is not None]
        if not lines:
            return None
        doc_vw = int(doc.get("viewport_width", VIEWPORT_WIDTH))
        vw = min(doc_vw, viewport_width)
        vh = int(doc.get("viewport_height", VIEWPORT_HEIGHT))
        color = str(doc.get("color") or LOGO_COLOR)
        style = f"bold {color} on #000000"
        token = str(doc.get("chomp_token") or _chomp_token_from_lines(lines))
        return StaticLogoArt(
            width=vw,
            height=vh,
            text=_center_lines_in_viewport(
                lines, viewport_width=vw, viewport_height=vh, style=style
            ),
            chomp_token=token,
        )
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None
