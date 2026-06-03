"""Unit tests for agent backend selection (celery default, hatchet for allowlisted)."""

from app.modules.intelligence.agents.runtime import backend_selection as bs

_FLAGS = ("AGENT_TASK_BACKEND", "HATCHET_AGENT_ALLOWLIST")


def _clear(monkeypatch):
    for k in _FLAGS:
        monkeypatch.delenv(k, raising=False)


def test_defaults_to_celery(monkeypatch):
    _clear(monkeypatch)
    assert bs.select_backend("debugging_agent") == "celery"
    assert bs.select_backend("codebase_qna_agent") == "celery"


def test_celery_default_does_not_route_allowlisted_agent(monkeypatch):
    _clear(monkeypatch)
    # Even the allowlisted agent stays on celery until hatchet-mode is turned on.
    assert bs.select_backend("debugging_agent") == "celery"


def test_hatchet_for_allowlisted_when_backend_is_hatchet(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENT_TASK_BACKEND", "hatchet")
    assert bs.select_backend("debugging_agent") == "hatchet"


def test_non_allowlisted_stays_celery_in_hatchet_mode(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENT_TASK_BACKEND", "hatchet")
    assert bs.select_backend("codebase_qna_agent") == "celery"


def test_custom_allowlist_parsing_with_whitespace(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENT_TASK_BACKEND", "hatchet")
    monkeypatch.setenv("HATCHET_AGENT_ALLOWLIST", " foo_agent , bar_agent ")
    assert bs.select_backend("foo_agent") == "hatchet"
    assert bs.select_backend("bar_agent") == "hatchet"
    # debugging_agent no longer allowlisted once the list is overridden
    assert bs.select_backend("debugging_agent") == "celery"


def test_none_or_empty_agent_id_is_celery(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENT_TASK_BACKEND", "hatchet")
    assert bs.select_backend(None) == "celery"
    assert bs.select_backend("") == "celery"
