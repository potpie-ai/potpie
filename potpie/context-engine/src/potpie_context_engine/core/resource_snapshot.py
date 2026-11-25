"""The portable UTF-8 file portion of a graph snapshot bundle."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from potpie_context_engine.core.ports.resource_store import (
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
            not parts
            or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name
            or "\\" in name
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
        if tail not in {("meta.json",), ("elements.jsonl",)}:
            if len(tail) != 2:
                raise ValueError(f"invalid resource snapshot path: {name!r}")
            require_resource_slug(tail[0], kind="section")
            filename = tail[1]
            suffix = next(
                (
                    suffix
                    for suffix in (".prov.json", ".ocr.txt", ".txt")
                    if filename.endswith(suffix)
                ),
                None,
            )
            seq = filename.removesuffix(suffix) if suffix else ""
            if (
                suffix is None
                or not seq.isascii()
                or not seq.isdigit()
                or f"{int(seq):04d}{suffix}" != filename
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
    normalized = {
        "format_version": RESOURCE_SNAPSHOT_VERSION,
        "files": dict(sorted(result.items())),
    }
    binary = payload.get("binary_files", {})
    if not isinstance(binary, Mapping) or len(binary) > 100_000:
        raise ValueError("snapshot binary_files must be an object")
    safe_binary: dict[str, str] = {}
    for name, encoded in binary.items():
        if not isinstance(name, str) or not isinstance(encoded, str):
            raise ValueError("binary snapshot paths and contents must be strings")
        parts = PurePosixPath(name).parts
        if (
            PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name
            or "\\" in name
            or any(part in {".", ".."} for part in parts)
            or len(parts) < 3
        ):
            raise ValueError(f"unsafe binary resource snapshot path: {name!r}")
        require_resource_slug(parts[0], kind="document")
        tail = parts[1:]
        if tail and tail[0] == ".versions":
            if len(tail) < 4 or not tail[1].isdigit() or int(tail[1]) < 1:
                raise ValueError(f"invalid binary resource revision: {name!r}")
            tail = tail[2:]
        if (
            len(tail) != 2
            or tail[0] != "artifacts"
            or not tail[1]
            or tail[1].startswith(".")
        ):
            raise ValueError(f"invalid binary resource snapshot path: {name!r}")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("binary resource snapshot is not valid base64") from exc
        total += len(decoded)
        if total > RESOURCE_IMPORT_MAX_BYTES:
            raise ValueError(
                f"resource snapshot exceeds {RESOURCE_IMPORT_MAX_BYTES} bytes"
            )
        safe_binary[name] = encoded
    if safe_binary:
        normalized["binary_files"] = dict(sorted(safe_binary.items()))
    return normalized
