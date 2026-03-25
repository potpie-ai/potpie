"""Utilities to group flat GitHub review comments into threads."""

from collections import defaultdict
from typing import Any


def _sort_key(comment: dict[str, Any]) -> tuple[str, str]:
    created_at = str(comment.get("created_at") or "")
    cid = str(comment.get("id") or "")
    return (created_at, cid)


def group_review_threads(flat_comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not flat_comments:
        return []

    by_id: dict[Any, dict[str, Any]] = {}
    for comment in flat_comments:
        cid = comment.get("id")
        if cid is not None:
            by_id[cid] = comment

    def find_root(comment: dict[str, Any]) -> Any:
        seen = set()
        current = comment
        root = current.get("id")
        while current.get("in_reply_to_id") is not None:
            parent_id = current.get("in_reply_to_id")
            if parent_id in seen:
                break
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                root = parent_id
                break
            root = parent.get("id", root)
            current = parent
        return root

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for comment in flat_comments:
        grouped[find_root(comment)].append(comment)

    threads: list[dict[str, Any]] = []
    for root_id, comments in grouped.items():
        sorted_comments = sorted(comments, key=_sort_key)
        lead = sorted_comments[0]

        path = next((c.get("path") for c in sorted_comments if c.get("path")), None)
        line = next((c.get("line") for c in sorted_comments if c.get("line") is not None), None)
        diff_hunk = next((c.get("diff_hunk") for c in sorted_comments if c.get("diff_hunk")), None)

        thread_comments = [
            {
                "id": c.get("id"),
                "author": ((c.get("user") or {}).get("login") if isinstance(c.get("user"), dict) else c.get("author")),
                "body": c.get("body") or "",
                "created_at": c.get("created_at"),
                "in_reply_to_id": c.get("in_reply_to_id"),
            }
            for c in sorted_comments
        ]

        threads.append(
            {
                "thread_id": root_id if root_id is not None else lead.get("id"),
                "path": path,
                "line": line,
                "diff_hunk": diff_hunk,
                "comments": thread_comments,
            }
        )

    return sorted(threads, key=lambda t: _sort_key({"created_at": t["comments"][0].get("created_at"), "id": t["thread_id"]}))
