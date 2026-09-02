"""Per-process memo for the read-only host calls the CLI repeats.

Every command resolves its pot, prints a pot header, and decides whether to
offer empty-pot guidance, and each of those steps used to ask the host again
for the same pot list, source list, and data-plane counts: eight RPCs for
``graph read`` and eighteen for ``graph status --json``, of which one was the
command's real call. In process each repeat is a dict read; against a hosted
control plane each is a full round trip, so the fan-out — not the graph — set
the floor for every command on a managed host.

One answer per ``(host, call, args)`` is kept for the life of the process. A
CLI invocation lasts a few hundred milliseconds, so staleness inside it can
only come from the process's own writes, and those clear the memo: injecting
a host, :func:`invalidate_host_snapshot` after every mutating pot/source
command, and the RPC client after any call the daemon declares non-read-only.

Two rules keep it honest. An exception is never cached — a call that raised is
asked again next time. And an entry is only reused for the *same* host object:
the host is held alongside its answers, so a fake built per test cannot inherit
a previous fake's answers through a recycled ``id()``.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, TypeVar

_T = TypeVar("_T")

_Key = tuple[int, str, tuple[Hashable, ...]]
_entries: dict[_Key, tuple[Any, Any]] = {}


def memoized(
    host: Any, call: str, args: tuple[Hashable, ...], fn: Callable[[], _T]
) -> _T:
    """``fn()``, answered from the memo when ``host`` already answered it.

    ``call`` names the host method (``"pots.list_pots"``) and ``args`` its
    arguments, so two pots' source lists never share an entry.
    """
    key = (id(host), call, args)
    entry = _entries.get(key)
    if entry is not None and entry[0] is host:
        return entry[1]
    value = fn()
    _entries[key] = (host, value)
    return value


def invalidate_host_snapshot() -> None:
    """Forget every memoized answer; the next read asks the host again."""
    _entries.clear()


def snapshot_entry_count() -> int:
    """How many answers are held — for tests that pin the memo's behaviour."""
    return len(_entries)


__all__ = ["invalidate_host_snapshot", "memoized", "snapshot_entry_count"]
