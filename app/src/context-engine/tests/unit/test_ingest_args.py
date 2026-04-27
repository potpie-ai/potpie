"""ingest CLI argument resolution."""

from pathlib import Path

import pytest

from adapters.inbound.cli import credentials_store as cs
from adapters.inbound.cli.ingest_args import (
    default_episode_name,
    looks_like_uuid,
    merge_file_body_into_ingest,
    resolve_ingest_body_and_pot,
)


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("550e8400-e29b-41d4-a716-446655440000", True),
        (" 550e8400-e29b-41d4-a716-446655440000 ", True),
        ("not-a-uuid", False),
        ("I like potpie", False),
    ],
)
def test_looks_like_uuid(s: str, expected: bool) -> None:
    assert looks_like_uuid(s) is expected


def test_one_positional_text() -> None:
    pot, body = resolve_ingest_body_and_pot("I like potpie", None, None)
    assert pot is None
    assert body == "I like potpie"


def test_one_positional_overridden_by_b() -> None:
    pot, body = resolve_ingest_body_and_pot("ignored", None, "from flag")
    assert pot is None
    assert body == "from flag"


def test_two_positionals() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    pot, body = resolve_ingest_body_and_pot(pid, "hello", None)
    assert pot == pid
    assert body == "hello"


def test_two_positionals_b_overrides_second() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    pot, body = resolve_ingest_body_and_pot(pid, "hello", "from flag")
    assert pot == pid
    assert body == "from flag"


def test_uuid_only_with_b() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    pot, body = resolve_ingest_body_and_pot(pid, None, "body text")
    assert pot == pid
    assert body == "body text"


def test_no_args_needs_b() -> None:
    with pytest.raises(ValueError, match="no_body"):
        resolve_ingest_body_and_pot(None, None, None)


def test_uuid_only_no_b() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    with pytest.raises(ValueError, match="uuid_needs_body"):
        resolve_ingest_body_and_pot(pid, None, None)


def test_two_args_first_not_uuid() -> None:
    with pytest.raises(ValueError, match="two_args_first_not_uuid"):
        resolve_ingest_body_and_pot("not-uuid", "hello", None)


def test_default_episode_name_first_line() -> None:
    assert default_episode_name("Hello\nworld") == "Hello"


def test_default_episode_name_truncates() -> None:
    long = "x" * 200
    out = default_episode_name(long)
    assert out.endswith("…")
    assert len(out) == 121


def test_merge_file_body_no_file() -> None:
    a, b = merge_file_body_into_ingest("hello", None, None, None)
    assert a == "hello" and b is None


def test_merge_file_body_with_uuid() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    a, b = merge_file_body_into_ingest(pid, None, None, "from file\n")
    assert a == pid and b == "from file\n"


def test_merge_file_body_conflict_b() -> None:
    with pytest.raises(ValueError, match="file_conflict_episode_body"):
        merge_file_body_into_ingest(None, None, "inline", "file")


def test_merge_file_body_conflict_second() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    with pytest.raises(ValueError, match="file_conflict_second"):
        merge_file_body_into_ingest(pid, "pos2", None, "file")


def test_merge_file_body_conflict_first_non_uuid() -> None:
    with pytest.raises(ValueError, match="file_conflict_first"):
        merge_file_body_into_ingest("not uuid", None, None, "file")


def test_merge_file_body_with_pot_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    uid = "44444444-4444-4444-4444-444444444444"
    cs.register_pot_alias("ws", uid)
    a, b = merge_file_body_into_ingest("ws", None, None, "from file\n")
    assert a == "ws" and b == "from file\n"
    pot, body = resolve_ingest_body_and_pot(a, None, b)
    assert pot == uid
    # Episode opt is `.strip()` end-to-end (trailing newline from file not preserved).
    assert body == "from file"


def test_two_positionals_with_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    uid = "55555555-5555-5555-5555-555555555555"
    cs.register_pot_alias("my-pot", uid)
    pot, body = resolve_ingest_body_and_pot("my-pot", "hello", None)
    assert pot == uid
    assert body == "hello"


def test_alias_with_episode_body_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    uid = "66666666-6666-6666-6666-666666666666"
    cs.register_pot_alias("p", uid)
    pot, body = resolve_ingest_body_and_pot("p", None, "body from -b")
    assert pot == uid
    assert body == "body from -b"
