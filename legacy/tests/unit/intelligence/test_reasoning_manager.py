from pathlib import Path

import pytest

from app.modules.intelligence.tools import reasoning_manager


pytestmark = pytest.mark.unit


def test_reasoning_manager_finalize_and_save_writes_hash_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = reasoning_manager.ReasoningManager()
    manager.append_content("alpha")
    manager.append_content(" beta")

    reasoning_hash = manager.finalize_and_save()

    assert reasoning_hash == manager.get_reasoning_hash()
    saved_path = tmp_path / ".data" / "reasoning" / f"{reasoning_hash}.txt"
    assert saved_path.read_text(encoding="utf-8") == "alpha beta"


def test_reasoning_manager_finalize_returns_none_when_empty():
    manager = reasoning_manager.ReasoningManager()

    assert manager.finalize_and_save() is None
    assert manager.get_reasoning_hash() is None


def test_get_and_reset_reasoning_manager_replaces_context_instance():
    reasoning_manager._reset_reasoning_manager()
    first = reasoning_manager._get_reasoning_manager()
    first.append_content("hello")

    same = reasoning_manager._get_reasoning_manager()
    reasoning_manager._reset_reasoning_manager()
    second = reasoning_manager._get_reasoning_manager()

    assert same is first
    assert second is not first
    assert second.content == ""
    assert second.get_reasoning_hash() is None
