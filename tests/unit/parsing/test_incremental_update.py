"""Unit tests for incremental knowledge-graph change detection."""

from unittest.mock import MagicMock

import pytest

from app.modules.parsing.graph_construction.incremental_update_service import (
    ChangeSet,
    FileSnapshot,
    IncrementalGraphUpdater,
    detect_changes,
)


pytestmark = pytest.mark.unit


def _hash(text: str) -> str:
    return FileSnapshot.hash_content(text)


class TestFileSnapshot:
    def test_hash_content_deterministic(self):
        assert FileSnapshot.hash_content("foo") == FileSnapshot.hash_content("foo")

    def test_hash_content_str_and_bytes_match(self):
        assert FileSnapshot.hash_content("foo") == FileSnapshot.hash_content(b"foo")

    def test_from_files_copies_mapping(self):
        src = {"a.py": _hash("x")}
        snap = FileSnapshot.from_files(src)
        src["b.py"] = _hash("y")
        # Original construction should not be mutated by later writes.
        assert "b.py" not in snap.files


class TestDetectChanges:
    def test_empty_to_empty_is_empty(self):
        cs = detect_changes(FileSnapshot(), FileSnapshot())
        assert cs.is_empty()
        assert cs.added == []
        assert cs.modified == []
        assert cs.deleted == []

    def test_added_files_detected(self):
        old = FileSnapshot.from_files({})
        new = FileSnapshot.from_files({"a.py": _hash("x"), "b.py": _hash("y")})
        cs = detect_changes(old, new)
        assert cs.added == ["a.py", "b.py"]
        assert cs.modified == []
        assert cs.deleted == []

    def test_deleted_files_detected(self):
        old = FileSnapshot.from_files({"a.py": _hash("x"), "b.py": _hash("y")})
        new = FileSnapshot.from_files({"a.py": _hash("x")})
        cs = detect_changes(old, new)
        assert cs.added == []
        assert cs.modified == []
        assert cs.deleted == ["b.py"]

    def test_modified_files_detected(self):
        old = FileSnapshot.from_files({"a.py": _hash("x"), "b.py": _hash("y")})
        new = FileSnapshot.from_files({"a.py": _hash("x"), "b.py": _hash("y2")})
        cs = detect_changes(old, new)
        assert cs.added == []
        assert cs.modified == ["b.py"]
        assert cs.deleted == []

    def test_mixed_changes(self):
        old = FileSnapshot.from_files(
            {"keep.py": _hash("k"), "change.py": _hash("v1"), "gone.py": _hash("g")}
        )
        new = FileSnapshot.from_files(
            {"keep.py": _hash("k"), "change.py": _hash("v2"), "new.py": _hash("n")}
        )
        cs = detect_changes(old, new)
        assert cs.added == ["new.py"]
        assert cs.modified == ["change.py"]
        assert cs.deleted == ["gone.py"]
        assert not cs.is_empty()

    def test_affected_and_upserted_partition(self):
        cs = ChangeSet(
            added=["new.py"], modified=["change.py"], deleted=["gone.py"]
        )
        # affected = modified + deleted (these need their old graph nodes removed)
        assert set(cs.affected_files) == {"change.py", "gone.py"}
        # upserted = added + modified (these need their new subgraph written)
        assert set(cs.upserted_files) == {"new.py", "change.py"}


class TestIncrementalGraphUpdaterCleanup:
    def _make_updater(self):
        service = MagicMock()
        session_cm = MagicMock()
        session = MagicMock()
        session_cm.__enter__.return_value = session
        session_cm.__exit__.return_value = False
        service.driver.session.return_value = session_cm
        return IncrementalGraphUpdater(service), service, session

    def test_cleanup_files_noop_on_empty(self):
        updater, _service, session = self._make_updater()
        assert updater.cleanup_files("proj", []) == 0
        session.run.assert_not_called()

    def test_cleanup_files_filters_blank_paths(self):
        updater, _service, session = self._make_updater()
        result_record = {"deleted": 3}

        run_result = MagicMock()
        run_result.single.return_value = result_record
        session.run.return_value = run_result

        deleted = updater.cleanup_files("proj", ["", "a.py", None, "b.py"])
        assert deleted == 3
        # The cypher should be called once with the cleaned list.
        called_args = session.run.call_args
        assert called_args.kwargs["paths"] == ["a.py", "b.py"]
        assert called_args.kwargs["project_id"] == "proj"

    def test_apply_changes_empty_changeset_short_circuits(self):
        updater, _service, session = self._make_updater()
        stats = updater.apply_changes(
            "proj", "user", ChangeSet(), new_artifacts=None
        )
        assert stats == {
            "deleted_nodes": 0,
            "inserted_nodes": 0,
            "inserted_edges": 0,
        }
        session.run.assert_not_called()

    def test_apply_changes_deletes_without_artifacts(self):
        updater, service, session = self._make_updater()
        run_result = MagicMock()
        run_result.single.return_value = {"deleted": 5}
        session.run.return_value = run_result

        cs = ChangeSet(modified=["a.py"], deleted=["b.py"])
        stats = updater.apply_changes("proj", "user", cs, new_artifacts=None)

        assert stats["deleted_nodes"] == 5
        assert stats["inserted_nodes"] == 0
        assert stats["inserted_edges"] == 0
        # Should not have attempted to write a subgraph back.
        service._store_graph.assert_not_called()
