"""Potpie CLI visual language helpers."""

import pytest
from rich.markup import escape

from potpie.context_engine.adapters.inbound.cli.ui.brand import LOGO_COLOR, UI_FOCUS_STYLE
from potpie.context_engine.adapters.inbound.cli.ui.format import (
    format_key_value_line,
    format_line,
    format_list_line,
    infer_line_tone,
    print_human_block,
    success_markup,
)

_USER_MARKUP = "[bold red]x[/bold red]"


def test_success_markup_includes_checkmark() -> None:
    assert "✓" in success_markup("active pot → foo")


def test_success_markup_escapes_user_markup() -> None:
    line = success_markup("[bold red]owned[/bold red]")
    assert "[bold red]owned[/bold red]" not in line
    assert escape("[bold red]owned[/bold red]") in line


def test_format_line_plain_escapes_user_markup() -> None:
    line = format_line("[green]pot[/green]", tone="plain")
    assert line == escape("[green]pot[/green]")


@pytest.mark.parametrize("tone", ["step", "muted", "title", "warn"])
def test_format_line_tone_escapes_user_markup(tone: str) -> None:
    line = format_line(_USER_MARKUP, tone=tone)
    assert _USER_MARKUP not in line
    assert escape(_USER_MARKUP) in line


def test_format_list_line_escapes_user_markup_in_pot_name() -> None:
    pot_name = "[bold red]evil-pot[/bold red]"
    line = format_list_line(f"* {pot_name} (id-1)")
    assert pot_name not in line
    assert escape(pot_name) in line


def test_format_key_value_line_escapes_user_markup() -> None:
    value = "[bold red]in_process[/bold red]"
    line = format_key_value_line(f"daemon: {value}")
    assert value not in line
    assert escape(value) in line
    assert f"[{LOGO_COLOR}]daemon:[/{LOGO_COLOR}]" in line


def test_infer_line_tone_success() -> None:
    assert infer_line_tone("Logged in to Potpie successfully.") == "success"
    assert infer_line_tone("active pot → foo") == "success"


def test_infer_line_tone_muted_empty() -> None:
    assert infer_line_tone("(no active pot)") == "muted"


def test_infer_line_tone_title() -> None:
    assert infer_line_tone("Linear workspaces:") == "title"


def test_format_line_step() -> None:
    line = format_line("Opening browser to sign in...", tone="step")
    assert "›" in line


def test_format_line_active_label_is_green() -> None:
    line = format_line("active: foo-pot (pot_527f9b3bbe92)")
    assert f"[{LOGO_COLOR}]active:[/{LOGO_COLOR}]" in line
    assert "foo-pot" in line
    assert "✓" not in line


def test_format_key_value_line_doctor_style() -> None:
    line = format_key_value_line("daemon: in_process (up)")
    assert f"[{LOGO_COLOR}]daemon:[/{LOGO_COLOR}]" in line
    assert f"[{UI_FOCUS_STYLE}]in_process (up)[/{UI_FOCUS_STYLE}]" in line


def test_print_human_block_doctor_lines(capsys, monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_FORCE_UI", "1")
    print_human_block(
        "daemon: in_process (up)\n"
        "backend: embedded caps=mutation, claim_query\n"
        "ledger: managed available=False"
    )
    out = capsys.readouterr().out
    assert "daemon:" in out
    assert "in_process" in out
    assert "embedded" in out


def test_print_human_block_single_line(capsys, monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_FORCE_UI", "1")
    print_human_block("active pot → demo-pot")
    out = capsys.readouterr().out
    assert "demo-pot" in out
    assert "✓" in out


def test_format_list_line_active_is_green_with_white_star() -> None:
    line = format_list_line("* alpha (id-1)")
    assert f"[{UI_FOCUS_STYLE}]*[/{UI_FOCUS_STYLE}]" in line
    assert f"[{LOGO_COLOR}]" in line
    assert "alpha (id-1)" in line


def test_format_list_line_inactive_is_white() -> None:
    line = format_list_line(" beta (id-2)")
    assert line.startswith(f"[{UI_FOCUS_STYLE}]")
    assert "beta (id-2)" in line
    assert LOGO_COLOR not in line


def test_print_human_block_list(capsys, monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_FORCE_UI", "1")
    print_human_block("* alpha (id-1)\n beta (id-2)")
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" in out


def test_print_human_block_section_title_and_list(capsys, monkeypatch) -> None:
    monkeypatch.setenv("POTPIE_FORCE_UI", "1")
    print_human_block("Local\n* alpha (id-1)\n beta (id-2)")
    out = capsys.readouterr().out
    assert "Local" in out
    assert "alpha" in out
    assert "beta" in out
