"""ingest CLI argument resolution."""

import pytest

from adapters.inbound.cli.ingest_args import (
    default_episode_name,
    looks_like_uuid,
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
