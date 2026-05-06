"""Group flat GitHub review comments into threads."""

from __future__ import annotations

import pytest

from domain.review_thread_grouper import group_review_threads

pytestmark = pytest.mark.unit


def _comment(
    cid: int,
    *,
    in_reply_to_id: int | None = None,
    user_login: str | None = "alice",
    body: str = "",
    created_at: str = "2026-04-27T00:00:00",
    path: str | None = None,
    line: int | None = None,
    diff_hunk: str | None = None,
) -> dict:
    out: dict = {
        "id": cid,
        "in_reply_to_id": in_reply_to_id,
        "body": body,
        "created_at": created_at,
    }
    if user_login is not None:
        out["user"] = {"login": user_login}
    if path is not None:
        out["path"] = path
    if line is not None:
        out["line"] = line
    if diff_hunk is not None:
        out["diff_hunk"] = diff_hunk
    return out


class TestGroupReviewThreads:
    def test_empty_input(self) -> None:
        assert group_review_threads([]) == []

    def test_single_root_comment_is_one_thread(self) -> None:
        threads = group_review_threads(
            [_comment(1, body="hi", path="a.py", line=10, diff_hunk="@@")]
        )
        assert len(threads) == 1
        thread = threads[0]
        assert thread["thread_id"] == 1
        assert thread["path"] == "a.py"
        assert thread["line"] == 10
        assert thread["diff_hunk"] == "@@"
        assert len(thread["comments"]) == 1
        assert thread["comments"][0]["author"] == "alice"
        assert thread["comments"][0]["body"] == "hi"

    def test_reply_chain_collapses_to_root(self) -> None:
        threads = group_review_threads(
            [
                _comment(1, body="root", path="a.py", line=3),
                _comment(2, in_reply_to_id=1, body="reply"),
                _comment(3, in_reply_to_id=2, body="reply2"),
            ]
        )
        assert len(threads) == 1
        thread = threads[0]
        assert thread["thread_id"] == 1
        # Comments sorted by created_at, then id; same timestamp here so id order.
        assert [c["id"] for c in thread["comments"]] == [1, 2, 3]

    def test_two_independent_threads_kept_separate(self) -> None:
        threads = group_review_threads(
            [
                _comment(1, body="root1", path="a.py"),
                _comment(2, in_reply_to_id=1, body="r1"),
                _comment(10, body="root2", path="b.py"),
                _comment(11, in_reply_to_id=10, body="r2"),
            ]
        )
        assert len(threads) == 2
        ids = sorted(t["thread_id"] for t in threads)
        assert ids == [1, 10]

    def test_orphaned_reply_uses_parent_id_as_root(self) -> None:
        # Reply references a missing parent; root falls back to the parent_id.
        threads = group_review_threads(
            [_comment(2, in_reply_to_id=999, body="orphan", path="x.py")]
        )
        assert len(threads) == 1
        assert threads[0]["thread_id"] == 999

    def test_cycle_in_reply_chain_terminates(self) -> None:
        # Comment 1 → 2 → 1 (illegal but we shouldn't infinite loop).
        # Each comment finds itself as its own root after the cycle breaks, so they
        # produce two threads — the important property is that find_root returns.
        threads = group_review_threads(
            [
                _comment(1, in_reply_to_id=2, body="a"),
                _comment(2, in_reply_to_id=1, body="b"),
            ]
        )
        # Doesn't infinite-loop; both comments accounted for.
        all_ids = sorted(
            c["id"] for t in threads for c in t["comments"]
        )
        assert all_ids == [1, 2]

    def test_missing_id_comments_preserved(self) -> None:
        # Comments without ids can't be looked up but shouldn't crash.
        threads = group_review_threads(
            [
                {"body": "hi", "created_at": "2026-04-27T00:00:00"},
            ]
        )
        # Falls into one thread keyed by None root → lead.id (None) used as thread_id.
        assert len(threads) == 1

    def test_thread_picks_first_path_line_diff_hunk(self) -> None:
        # Path/line/diff_hunk come from the earliest comment that has them.
        threads = group_review_threads(
            [
                _comment(1, body="root", path=None, line=None, diff_hunk=None),
                _comment(2, in_reply_to_id=1, body="r", path="late.py", line=99),
            ]
        )
        thread = threads[0]
        assert thread["path"] == "late.py"
        assert thread["line"] == 99

    def test_user_dict_extracts_login(self) -> None:
        threads = group_review_threads(
            [_comment(1, body="hi", user_login="bob")]
        )
        assert threads[0]["comments"][0]["author"] == "bob"

    def test_user_falls_back_to_author_field_when_not_dict(self) -> None:
        # When ``user`` is not a dict, fall back to ``author`` field directly.
        threads = group_review_threads(
            [
                {
                    "id": 1,
                    "user": "raw-string",
                    "author": "fallback-author",
                    "body": "hi",
                    "created_at": "2026-04-27",
                }
            ]
        )
        # The grouper checks isinstance(user, dict) — here user is a string, so author is used.
        assert threads[0]["comments"][0]["author"] == "fallback-author"

    def test_threads_sorted_by_first_comment_time(self) -> None:
        threads = group_review_threads(
            [
                _comment(1, body="early", created_at="2026-01-01"),
                _comment(10, body="late", created_at="2026-12-31"),
            ]
        )
        # Sort key is (created_at, id) of the first comment in each thread.
        assert [t["thread_id"] for t in threads] == [1, 10]

    def test_comment_sort_uses_created_at_then_id(self) -> None:
        threads = group_review_threads(
            [
                _comment(2, in_reply_to_id=1, body="reply", created_at="2026-04-28"),
                _comment(1, body="root", created_at="2026-04-27"),
            ]
        )
        assert [c["id"] for c in threads[0]["comments"]] == [1, 2]
