"""Bounded, time-filtered and followable reads of the daemon log.

The daemon log is append-only and nothing rotates it, so a whole-file read
grows for as long as the daemon has been in use. Reads here are bounded by
default: the tail walks backwards from the end of the file and stops once it
holds enough lines.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

#: How many trailing lines a log read returns when the caller does not say.
DEFAULT_LOG_TAIL_LINES: Final[int] = 200

#: How often a follow looks for new bytes: short enough to feel live, long
#: enough that tailing an idle daemon costs nothing.
FOLLOW_POLL_SECONDS: Final[float] = 0.4

#: The backwards tail read walks the file in chunks of this size.
_TAIL_CHUNK_BYTES: Final[int] = 64 * 1024

#: ``2026-08-12 11:21:33,123`` (stdlib ``asctime``) or an ISO-8601 ``T``.
_PLAIN_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:[.,](\d+))?"
)

#: ``--since 15m``: an age relative to now.
_RELATIVE_SINCE = re.compile(r"^(\d+)\s*([smhd])$", re.IGNORECASE)

_RELATIVE_SINCE_UNITS: Final[dict[str, str]] = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def parse_since(value: str) -> datetime:
    """Read ``--since`` as an ISO-8601 instant or a relative age (``15m``).

    Raises :class:`ValueError`, which the CLI error boundary renders as a
    validation failure naming the offending value.
    """

    text = value.strip()
    relative = _RELATIVE_SINCE.match(text)
    if relative is not None:
        amount = int(relative.group(1))
        unit = _RELATIVE_SINCE_UNITS[relative.group(2).lower()]
        return datetime.now() - timedelta(**{unit: amount})
    try:
        return _as_naive_local(datetime.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"cannot read {value!r} as a time; use an ISO-8601 timestamp "
            "(2026-08-12T09:00:00) or a relative age (30s, 15m, 2h, 1d)"
        ) from None


def read_log_tail(
    path: Path,
    *,
    tail: int | None = DEFAULT_LOG_TAIL_LINES,
    since: datetime | None = None,
) -> list[str]:
    """The last ``tail`` lines of ``path``, newest last.

    ``tail=None`` (or ``0``) reads the whole file, which is only ever an
    explicit request.
    """

    lines, _offset = _tail_with_offset(path, tail)
    return filter_since(lines, since) if since is not None else lines


def follow_log(
    locate: Callable[[], Path | None],
    *,
    tail: int | None = DEFAULT_LOG_TAIL_LINES,
    since: datetime | None = None,
    poll_interval: float = FOLLOW_POLL_SECONDS,
) -> Iterator[str]:
    """Yield the tail of the log, then every line appended after it, forever.

    ``locate`` names the current log file, or ``None`` while there is none, so
    a follow started before the daemon has written anything picks the file up
    once it appears. The stream polls rather than watching the filesystem, and
    restarts from the beginning of a file that was truncated or replaced.
    """

    log_file = locate()
    position = 0
    if log_file is not None:
        # The backlog and the resume offset come from one open, and the offset
        # is fixed before the first yield: lines appended while the consumer
        # handles the backlog land after the offset and are read next.
        try:
            backlog, position = _tail_with_offset(log_file, tail)
        except OSError:
            log_file = None
            backlog = []
        yield from backlog if since is None else filter_since(backlog, since)
    pending = ""
    while True:
        if log_file is None:
            log_file = locate()
            if log_file is None:
                time.sleep(poll_interval)
                continue
            position = 0
        try:
            size = log_file.stat().st_size
        except OSError:
            # Rediscover the file on the next pass; sleep first so a path that
            # exists but cannot be stat'd does not spin.
            log_file = None
            time.sleep(poll_interval)
            continue
        if size < position:
            position = 0
            pending = ""
        if size == position:
            time.sleep(poll_interval)
            continue
        with log_file.open("rb") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()
        pending += chunk.decode("utf-8", errors="replace")
        while "\n" in pending:
            line, _, pending = pending.partition("\n")
            line = line.removesuffix("\r")
            if since is None or _at_or_after(line, since):
                yield line


def filter_since(lines: list[str], since: datetime) -> list[str]:
    """Drop lines older than ``since``, keeping each kept line's undated tail.

    A traceback is one log event spread over many lines and only the first
    carries a timestamp, so an undated line inherits the verdict of the last
    dated line above it.
    """

    kept: list[str] = []
    including = False
    for line in lines:
        stamp = _line_timestamp(line)
        if stamp is not None:
            including = stamp >= since
        if including:
            kept.append(line)
    return kept


def _tail_with_offset(path: Path, tail: int | None) -> tuple[list[str], int]:
    """The tail, plus the byte offset it ends at, from a single open."""

    with path.open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        if tail is None or tail <= 0:
            handle.seek(0)
            data = handle.read(end)
        else:
            position = end
            data = b""
            # ``<=`` rather than ``<``: the last line usually ends in a
            # newline, so N newlines bound only N-1 whole lines.
            while position > 0 and data.count(b"\n") <= tail:
                step = min(_TAIL_CHUNK_BYTES, position)
                position -= step
                handle.seek(position)
                data = handle.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    return (lines if tail is None or tail <= 0 else lines[-tail:]), end


def _at_or_after(line: str, since: datetime) -> bool:
    stamp = _line_timestamp(line)
    # An undated line in a live stream follows a line that was already
    # admitted, so it belongs to the stream.
    return True if stamp is None else stamp >= since


def _line_timestamp(line: str) -> datetime | None:
    """The instant a log line carries, in either configured log format."""

    text = line.lstrip()
    if text.startswith("{"):
        try:
            stamp = json.loads(text).get("ts")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(stamp, str):
            return None
        try:
            return _as_naive_local(datetime.fromisoformat(stamp))
        except ValueError:
            return None
    match = _PLAIN_TIMESTAMP.match(text)
    if match is None:
        return None
    fraction = (match.group(3) or "0").ljust(6, "0")[:6]
    try:
        return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}.{fraction}")
    except ValueError:
        return None


def _as_naive_local(value: datetime) -> datetime:
    """Compare timestamps in one frame: this machine's wall clock.

    The plain formatter writes naive local time and the JSON formatter writes
    an offset, so a mixed log would otherwise fail on the first comparison.
    """

    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


__all__ = [
    "DEFAULT_LOG_TAIL_LINES",
    "FOLLOW_POLL_SECONDS",
    "filter_since",
    "follow_log",
    "parse_since",
    "read_log_tail",
]
