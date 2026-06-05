"""Interactive TTY prompts (arrow-key menus) for the host CLI."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.text import Text

from adapters.inbound.cli.ui.brand import (
    LOGO_STYLE,
    LOGO_SUBTLE_SEPARATOR_STYLE,
    UI_FOCUS_STYLE,
    UI_MUTED_STYLE,
)
from adapters.inbound.cli.ui.potpie_logo_anim import content_width_for_panel, panel_width_for_console
from adapters.inbound.cli.ui.setup_wizard_ui import is_interactive_tty

_SUBTLE_SEPARATOR_CHAR = "─"


def _stderr_console() -> Console:
    return Console(stderr=True)


def _option_line(*, focused: bool, label: str, chosen: bool = False) -> Text:
    prefix = "> " if (focused or chosen) else "  "
    if chosen:
        style = LOGO_STYLE
    elif focused:
        style = UI_FOCUS_STYLE
    else:
        style = UI_MUTED_STYLE
    return Text.assemble((prefix, style), (label, style))


def _checkbox_line(*, focused: bool, checked: bool, label: str) -> Text:
    box = "[x]" if checked else "[ ]"
    prefix = "> " if focused else "  "
    if checked:
        style = LOGO_STYLE
    elif focused:
        style = UI_FOCUS_STYLE
    else:
        style = UI_MUTED_STYLE
    return Text.assemble((prefix, style), (box, style), (" ", style), (label, style))


def _read_key() -> str:
    """Read one navigation key; returns up, down, enter, yes, no, or escape."""
    if not sys.stdin.isatty():
        return "enter"

    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch in ("y", "Y"):
            return "yes"
        if ch in ("n", "N"):
            return "no"
        if ch == "\x00" or ch == "\xe0":
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "up"
            if ch2 == "P":
                return "down"
        return "escape"

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == "\x03":
            raise KeyboardInterrupt
        if ch1 in ("\r", "\n"):
            return "enter"
        if ch1 == " ":
            return "space"
        if ch1 in ("y", "Y"):
            return "yes"
        if ch1 in ("n", "N"):
            return "no"
        if ch1 == "\x1b":
            rest = sys.stdin.read(2)
            if rest == "[A":
                return "up"
            if rest == "[B":
                return "down"
            return "escape"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "escape"


def _cursor_up_and_clear_lines(count: int) -> None:
    if count <= 0:
        return
    sys.stderr.write(f"\033[{count}A")
    for _ in range(count):
        sys.stderr.write("\033[2K")


def _draw_options(
    console: Console,
    *,
    selected: int,
    yes_label: str,
    no_label: str,
    repaint: bool,
    chosen_index: int | None = None,
) -> None:
    if repaint:
        _cursor_up_and_clear_lines(2)
    console.print(
        _option_line(
            focused=selected == 0 and chosen_index is None,
            label=yes_label,
            chosen=chosen_index == 0,
        )
    )
    console.print(
        _option_line(
            focused=selected == 1 and chosen_index is None,
            label=no_label,
            chosen=chosen_index == 1,
        )
    )


def prompt_yes_no(
    message: str,
    *,
    default: bool = True,
    yes_label: str = "Yes",
    no_label: str = "No",
) -> bool:
    """Arrow-key yes/no menu; focus is white, chosen row is green."""
    if not is_interactive_tty():
        import typer

        return typer.confirm(message, default=default)

    selected = 0 if default else 1
    console = _stderr_console()

    console.print()
    console.print(message)
    _draw_options(
        console, selected=selected, yes_label=yes_label, no_label=no_label, repaint=False
    )

    try:
        sys.stderr.write("\033[?25l")
        while True:
            key = _read_key()
            if key == "up":
                selected = (selected - 1) % 2
                _draw_options(
                    console,
                    selected=selected,
                    yes_label=yes_label,
                    no_label=no_label,
                    repaint=True,
                )
            elif key == "down":
                selected = (selected + 1) % 2
                _draw_options(
                    console,
                    selected=selected,
                    yes_label=yes_label,
                    no_label=no_label,
                    repaint=True,
                )
            elif key == "yes":
                _draw_options(
                    console,
                    selected=0,
                    yes_label=yes_label,
                    no_label=no_label,
                    repaint=True,
                    chosen_index=0,
                )
                return True
            elif key == "no":
                _draw_options(
                    console,
                    selected=1,
                    yes_label=yes_label,
                    no_label=no_label,
                    repaint=True,
                    chosen_index=1,
                )
                return False
            elif key == "enter":
                _draw_options(
                    console,
                    selected=selected,
                    yes_label=yes_label,
                    no_label=no_label,
                    repaint=True,
                    chosen_index=selected,
                )
                return selected == 0
            elif key == "escape":
                return default
    except KeyboardInterrupt:
        raise
    finally:
        sys.stderr.write("\033[?25h")
        console.print()


def _draw_checkboxes(
    console: Console,
    *,
    options: list[tuple[str, str]],
    focused: int,
    checked: set[str],
    repaint: bool,
    confirming: bool = False,
) -> None:
    if repaint:
        _cursor_up_and_clear_lines(len(options))
    for index, (option_id, label) in enumerate(options):
        console.print(
            _checkbox_line(
                focused=False if confirming else index == focused,
                checked=option_id in checked,
                label=label,
            )
        )


def prompt_multi_checkbox(
    message: str,
    options: list[tuple[str, str]],
    *,
    default_checked: frozenset[str] | None = None,
) -> list[str]:
    """Arrow-key multi-select; Space toggles, Enter confirms."""
    if not options:
        return []

    if not is_interactive_tty():
        import typer

        selected: list[str] = []
        for option_id, label in options:
            if typer.confirm(f"{label}?", default=option_id in (default_checked or frozenset())):
                selected.append(option_id)
        return selected

    checked: set[str] = set(default_checked or ())
    focused = 0
    console = _stderr_console()

    console.print()
    console.print(message)
    console.print(
        Text(
            "↑/↓ move · Space to select · Enter to continue",
            style=UI_MUTED_STYLE,
        )
    )
    _draw_checkboxes(
        console,
        options=options,
        focused=focused,
        checked=checked,
        repaint=False,
    )

    try:
        sys.stderr.write("\033[?25l")
        while True:
            key = _read_key()
            option_id = options[focused][0]
            if key == "up":
                focused = (focused - 1) % len(options)
                _draw_checkboxes(
                    console,
                    options=options,
                    focused=focused,
                    checked=checked,
                    repaint=True,
                )
            elif key == "down":
                focused = (focused + 1) % len(options)
                _draw_checkboxes(
                    console,
                    options=options,
                    focused=focused,
                    checked=checked,
                    repaint=True,
                )
            elif key == "enter":
                _draw_checkboxes(
                    console,
                    options=options,
                    focused=focused,
                    checked=checked,
                    repaint=True,
                    confirming=True,
                )
                return [option_id for option_id, _ in options if option_id in checked]
            elif key == "space":
                if option_id in checked:
                    checked.remove(option_id)
                else:
                    checked.add(option_id)
                _draw_checkboxes(
                    console,
                    options=options,
                    focused=focused,
                    checked=checked,
                    repaint=True,
                )
            elif key == "escape":
                return []
    except KeyboardInterrupt:
        raise
    finally:
        sys.stderr.write("\033[?25h")
        console.print()


def prompt_text(message: str, *, default: str = "") -> str:
    """Simple text prompt (restored TTY) for names and short answers."""
    import typer

    console = _stderr_console()
    console.print()
    console.print(message, style=LOGO_STYLE)
    return typer.prompt("  Name", default=default or None)


def _print_subtle_separator(console: Console) -> None:
    width = content_width_for_panel(panel_width_for_console(console))
    console.print(
        Text(_SUBTLE_SEPARATOR_CHAR * max(24, width), style=LOGO_SUBTLE_SEPARATOR_STYLE)
    )


def prompt_first_pot_name(*, default: str = "foo-pot") -> str:
    """Friendly post-setup prompt to name the user's first pot."""
    import typer

    console = _stderr_console()
    console.print()
    _print_subtle_separator(console)
    console.print()
    console.print(Text("Hey — setup is complete!", style=LOGO_STYLE))
    console.print(
        Text(
            "Let's start by creating your first pot (your workspace for this project).",
            style="white",
        )
    )
    console.print(Text("What would you like to call it?", style="white"))
    console.print()
    return typer.prompt("  Pot name", default=default or None)


__all__ = ["prompt_yes_no", "prompt_multi_checkbox", "prompt_text", "prompt_first_pot_name"]
