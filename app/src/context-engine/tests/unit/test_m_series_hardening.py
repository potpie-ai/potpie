"""M-1 / M-3 / H-3 hardening invariants."""

from __future__ import annotations

import pytest

from domain.error_redaction import redact_secrets, safe_error

pytestmark = pytest.mark.unit


def test_redacts_tokenized_clone_url():
    s = "fatal: unable to access 'https://x-access-token:ghp_AbCdEf0123456789abcd@github.com/o/r.git/'"
    out = redact_secrets(s)
    assert "ghp_AbCdEf0123456789abcd" not in out
    assert "x-access-token:" not in out or ":***@" in out
    assert "github.com/o/r.git" in out  # non-secret context preserved


def test_redacts_authorization_and_pat():
    assert "ghp_" not in redact_secrets("token=ghp_zzzzzzzzzzzzzzzzzzzzzz")
    out = redact_secrets("Authorization: Bearer sk-secret-value")
    assert "sk-secret-value" not in out
    assert "github_pat_" not in redact_secrets(
        "github_pat_11ABCDEFG0123456789_abcdefghij"
    )


def test_safe_error_is_bounded_and_redacted():
    msg = safe_error(Exception("https://u:p@h " + "x" * 999), limit=120)
    assert len(msg) <= 120
    assert "://u:p@h" not in msg


def test_canonical_writer_rejects_bad_pot_id():
    from adapters.outbound.graphiti.canonical_writer import (
        _require_valid_pot_id,
    )

    _require_valid_pot_id("550e8400-e29b-41d4-a716-446655440000")  # ok
    _require_valid_pot_id("pot_abc-123")  # ok
    for bad in ["", "  ", "a/b", "a b", "a;b", "a$b", "../x"]:
        with pytest.raises(ValueError):
            _require_valid_pot_id(bad)


def test_predicate_allowlist_blocks_injected_relations():
    """Rebuild plan P0: edge writes are :RELATES_TO {name: <predicate>}.

    The MERGE template fixes the relationship type to ``:RELATES_TO``;
    the only injected value is the predicate ``name``, which the writer
    requires to be a canonical-vocab entry.
    """
    from adapters.outbound.graphiti.canonical_writer import (
        _is_valid_predicate,
    )

    assert _is_valid_predicate("OWNS") is True
    assert _is_valid_predicate("Totally_Made_Up_Edge") is False
    assert _is_valid_predicate("bad-char") is False
    assert _is_valid_predicate("") is False


# NOTE: the provider_host allowlist (M-2) lives in the host module
# (app.modules.context_graph.attach_repo_to_pot) which is not importable
# from the isolated context-engine test rootdir; it is covered by the host
# suite / route-level ValueError → HTTP 400 path.
