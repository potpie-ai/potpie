"""Readable resource snapshots with staged, rollback-safe restore."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from potpie_context_core.resource_snapshot import normalize_resource_snapshot
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    META_FILENAME,
    VERSIONS_DIRNAME,
    LocalResourceStore,
    _load_manifest,
    _pot_lock,
    _read_revision_counter,
    _write_revision_counter,
    read_source_document,
)

logger = logging.getLogger(__name__)


def _read_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    if not root.exists():
        return files
    for document in sorted(root.iterdir()):
        if document.name.startswith("."):
            continue
        if document.is_symlink() or not document.is_dir():
            raise ValueError(f"unexpected resource store entry: {document.name}")
        for path in sorted(document.rglob("*")):
            if path.is_symlink():
                raise ValueError("resource snapshot cannot include symlinks")
            if path.is_file():
                with path.open(encoding="utf-8", newline="") as handle:
                    files[path.relative_to(root).as_posix()] = handle.read()
    return files


def export_resources(store: LocalResourceStore, *, pot_id: str) -> dict[str, Any]:
    root = store._pot_root(pot_id)
    with _pot_lock(root, exclusive=True):
        payload = normalize_resource_snapshot(
            {"format_version": "1", "files": _read_files(root)}
        )
        _validate_documents(root, payload["files"], pot_id=pot_id)
        return payload


def _validate_documents(root: Path, files: Mapping[str, str], *, pot_id: str) -> None:
    documents = {name.split("/", 1)[0] for name in files}
    expected: set[str] = set()
    for doc in documents:
        if f"{doc}/{META_FILENAME}" not in files:
            raise ValueError(f"snapshot document {doc!r} has no current manifest")
    for name in files:
        if not name.endswith("/meta.json"):
            continue
        directory = root / Path(name).parent
        source = read_source_document(directory)
        doc = name.split("/", 1)[0]
        manifest = _load_manifest(directory, pot_id=pot_id, doc=doc)
        if manifest is None or manifest.revision < 1:
            raise ValueError(f"invalid stored resource revision in {name}")
        raw = json.loads(files[name])
        if isinstance(raw.get("revision"), bool) or raw.get("revision") != manifest.revision:
            raise ValueError(f"invalid stored resource revision in {name}")
        if raw.get("doc") != doc:
            raise ValueError(f"resource document identity disagrees with {name}")
        parts = Path(name).parts
        if VERSIONS_DIRNAME in parts and int(parts[2]) != manifest.revision:
            raise ValueError(f"resource revision identity disagrees with {name}")
        prefix = Path(name).parent.as_posix()
        expected.add(name)
        expected.update(
            f"{prefix}/{section.slug}/{ref.seq:04d}.txt"
            for section in source.sections for ref in section.chunks
        )
    if expected != set(files):
        raise ValueError("resource snapshot contains files absent from its manifests")


@contextmanager
def restore_resources(
    store: LocalResourceStore, *, pot_id: str, payload: Mapping[str, Any]
) -> Iterator[None]:
    """Publish validated bytes before graph commit; roll back on graph failure.

    Existing documents must be byte-identical. A process crash before the graph
    commit may leave additional unreferenced files, never missing old evidence.
    """
    payload = normalize_resource_snapshot(payload)
    incoming = payload["files"]
    root = store._pot_root(pot_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-stage-", dir=root.parent))
    published: list[Path] = []
    try:
        for name, text in incoming.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        _validate_documents(staging, incoming, pot_id=pot_id)
        with _pot_lock(root, exclusive=True):
            existing = _read_files(root)
            incoming_docs = {name.split("/", 1)[0] for name in incoming}
            existing_docs = {name.split("/", 1)[0] for name in existing}
            for doc in incoming_docs & existing_docs:
                prefix = f"{doc}/"
                before = {k: v for k, v in existing.items() if k.startswith(prefix)}
                after = {k: v for k, v in incoming.items() if k.startswith(prefix)}
                if before != after:
                    raise ValueError(f"snapshot document conflicts with target: {doc}")
            if incoming_docs <= existing_docs:
                yield
                return
            # Reserve monotonically increasing revision counters even if the
            # graph rejects the restore; advancing a counter cannot reuse an ID.
            for doc in incoming_docs - existing_docs:
                revision = json.loads(incoming[f"{doc}/meta.json"])["revision"]
                _write_revision_counter(
                    root, doc, max(revision, _read_revision_counter(root, doc))
                )
            try:
                # Publish only new document directories. Existing evidence is
                # never moved out of place, including if this process exits
                # between renames. Readers share the pot lock and cannot see
                # a partial publication during a successful restore.
                root.mkdir(parents=True, exist_ok=True)
                for doc in sorted(incoming_docs - existing_docs):
                    destination = root / doc
                    os.replace(staging / doc, destination)
                    published.append(destination)
                yield
            except BaseException:
                for destination in reversed(published):
                    try:
                        shutil.rmtree(destination)
                    except OSError:
                        logger.exception("Could not remove uncommitted snapshot document: %s", destination)
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
