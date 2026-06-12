"""Potpie CLI visual language helpers."""

from adapters.inbound.cli.ui.brand import LOGO_COLOR, UI_FOCUS_STYLE
from adapters.inbound.cli.ui.format import (
    format_line,
    format_list_line,
    infer_line_tone,
    print_human_block,
    success_markup,
)


def test_success_markup_includes_checkmark() -> None:
    assert "✓" in success_markup("active pot → foo")


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
    from adapters.inbound.cli.ui.format import format_key_value_line

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
