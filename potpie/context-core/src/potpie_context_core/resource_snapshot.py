"""The portable UTF-8 file portion of a graph snapshot bundle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from potpie_context_core.ports.resource_store import (
    RESOURCE_IMPORT_MAX_BYTES,
    require_resource_slug,
)

RESOURCE_SNAPSHOT_VERSION = "1"


def normalize_resource_snapshot(payload: Any) -> dict[str, Any]:
    """Validate every name and byte before a snapshot creates files."""
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot resources must be an object")
    if payload.get("format_version") != RESOURCE_SNAPSHOT_VERSION:
        raise ValueError("unsupported resource snapshot format_version")
    files = payload.get("files")
    if not isinstance(files, Mapping) or len(files) > 100_000:
        raise ValueError("snapshot resource files must contain at most 100000 files")
    result: dict[str, str] = {}
    total = 0
    for name, text in files.items():
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("resource snapshot paths and contents must be strings")
        parts = PurePosixPath(name).parts
        if (
            not parts or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name or "\\" in name
            or any(part in {".", ".."} for part in parts)
        ):
            raise ValueError(f"unsafe resource snapshot path: {name!r}")
        require_resource_slug(parts[0], kind="document")
        tail = parts[1:]
        if tail and tail[0] == ".versions":
            if len(tail) < 3 or not tail[1].isascii() or not tail[1].isdigit():
                raise ValueError(f"invalid resource revision path: {name!r}")
            if int(tail[1]) < 1 or str(int(tail[1])) != tail[1]:
                raise ValueError(f"invalid resource revision: {name!r}")
            tail = tail[2:]
        if tail != ("meta.json",):
            if len(tail) != 2:
                raise ValueError(f"invalid resource snapshot path: {name!r}")
            require_resource_slug(tail[0], kind="section")
            seq = tail[1].removesuffix(".txt")
            if (
                not tail[1].endswith(".txt") or not seq.isascii() or not seq.isdigit()
                or f"{int(seq):04d}.txt" != tail[1]
            ):
                raise ValueError(f"invalid resource chunk filename: {name!r}")
        try:
            total += len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("resource snapshot text must be valid UTF-8") from exc
        if total > RESOURCE_IMPORT_MAX_BYTES:
            raise ValueError(
                f"resource snapshot exceeds {RESOURCE_IMPORT_MAX_BYTES} bytes"
            )
        result[name] = text
    return {"format_version": RESOURCE_SNAPSHOT_VERSION, "files": dict(sorted(result.items()))}
