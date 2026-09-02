"""``files`` is the form of an import that crosses the wire.

A host — a daemon with another working directory, or a managed service on
another machine — never sees the caller's chunk directory. These tests pin the
two halves of the transport: reading a directory into ``files`` on the caller's
side, and materialising ``files`` back into a directory on the host's side,
with the keys treated as untrusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from potpie_context_core.ports import resource_store as module
from potpie_context_core.ports.resource_store import (
    RESOURCE_IMPORT_INVALID,
    RESOURCE_MANIFEST_INVALID,
    ResourceStoreError,
    import_source,
    read_import_files,
)


def _write_directory(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps({"sections": [{"slug": "body", "chunks": [{"seq": 0}]}]}),
        encoding="utf-8",
    )
    (root / "body").mkdir()
    (root / "body" / "0000.txt").write_text("alpha\n\nomega", encoding="utf-8")
    return root


def test_read_import_files_keys_every_file_by_posix_relative_path(tmp_path):
    files = read_import_files(_write_directory(tmp_path / "out"))

    assert set(files) == {"meta.json", "body/0000.txt"}
    assert files["body/0000.txt"] == "alpha\n\nomega"


def test_read_import_files_skips_hidden_entries(tmp_path):
    root = _write_directory(tmp_path / "out")
    (root / ".DS_Store").write_bytes(b"\x00\x01")
    (root / ".swp").mkdir()
    (root / ".swp" / "0000.txt").write_text("stale", encoding="utf-8")

    assert set(read_import_files(root)) == {"meta.json", "body/0000.txt"}


def test_read_import_files_refuses_a_file_that_is_not_utf8(tmp_path):
    root = _write_directory(tmp_path / "out")
    (root / "body" / "0001.txt").write_bytes(b"\xff\xfe not text")

    with pytest.raises(ResourceStoreError) as excinfo:
        read_import_files(root)
    assert excinfo.value.code == RESOURCE_IMPORT_INVALID
    assert "body/0001.txt" in str(excinfo.value)


def test_read_import_files_refuses_a_missing_or_empty_directory(tmp_path):
    # Same code as a missing meta.json: the agent's repair is the same.
    with pytest.raises(ResourceStoreError) as missing:
        read_import_files(tmp_path / "nowhere")
    assert missing.value.code == RESOURCE_MANIFEST_INVALID

    (tmp_path / "empty").mkdir()
    with pytest.raises(ResourceStoreError) as empty:
        read_import_files(tmp_path / "empty")
    assert empty.value.code == RESOURCE_MANIFEST_INVALID


def test_read_import_files_refuses_a_tree_over_the_byte_cap(tmp_path, monkeypatch):
    root = _write_directory(tmp_path / "out")
    monkeypatch.setattr(module, "RESOURCE_IMPORT_MAX_BYTES", 8)

    with pytest.raises(ResourceStoreError) as excinfo:
        read_import_files(root)
    assert excinfo.value.code == RESOURCE_IMPORT_INVALID
    assert "exceeds" in str(excinfo.value)


def test_import_source_round_trips_files_through_a_scratch_directory(tmp_path):
    files = read_import_files(_write_directory(tmp_path / "out"))

    with import_source(None, files) as root:
        scratch = root
        assert (root / "meta.json").read_text(encoding="utf-8") == files["meta.json"]
        assert (root / "body" / "0000.txt").read_text(
            encoding="utf-8"
        ) == "alpha\n\nomega"
        assert root != tmp_path / "out"
    assert not scratch.exists(), "the scratch directory must not outlive the block"


def test_import_source_hands_a_path_straight_through(tmp_path):
    root = _write_directory(tmp_path / "out")

    with import_source(root, None) as given:
        assert given == root
    assert root.exists()


def test_import_source_needs_exactly_one_form(tmp_path):
    with pytest.raises(ResourceStoreError) as neither:
        with import_source(None, None):
            pass
    assert neither.value.code == RESOURCE_IMPORT_INVALID

    with pytest.raises(ResourceStoreError) as both:
        with import_source(tmp_path, {"meta.json": "{}"}):
            pass
    assert both.value.code == RESOURCE_IMPORT_INVALID


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/etc/passwd",
        "body/../../x.txt",
        "",
        " meta.json",
        "body\\0000.txt",
    ],
)
def test_import_source_refuses_a_key_that_leaves_the_scratch_directory(name, tmp_path):
    files = {"meta.json": "{}", name: "payload"}

    with pytest.raises(ResourceStoreError) as excinfo:
        with import_source(None, files):
            pass
    assert excinfo.value.code == RESOURCE_IMPORT_INVALID
    assert name.strip() in str(excinfo.value) or "invalid file name" in str(
        excinfo.value
    )
    assert not (tmp_path / "escape.txt").exists()


def test_import_source_refuses_a_non_text_value():
    with pytest.raises(ResourceStoreError) as excinfo:
        with import_source(None, {"meta.json": b"{}"}):  # type: ignore[dict-item]
            pass
    assert excinfo.value.code == RESOURCE_IMPORT_INVALID
