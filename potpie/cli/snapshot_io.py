"""Client-side files for portable graph snapshots.

The graph host exchanges JSON-compatible data only.  Paths belong to the CLI
process so a managed daemon never needs to see (or be able to reach) the
caller's filesystem.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from potpie_context_core.graph_snapshot import normalize_snapshot_payload
from potpie_context_core.resource_snapshot import normalize_resource_snapshot


class SnapshotIOError(ValueError):
    """A snapshot path or payload is incomplete, invalid, or unsafe to use."""


_README = """# Potpie graph snapshot

This folder is a portable, human-readable Potpie graph snapshot.

- `manifest.json` describes the format and names the data files.
- `entities.json` contains entity keys, labels, and properties.
- `claims.json` contains canonical graph claims.
- `resources/` contains document text when resources were included.

Import it with `potpie graph import <this-folder> --pot <destination>`.
Edit JSON carefully: import validates the complete folder before contacting the host.
"""


def local_path(raw: str) -> Path:
    """Resolve a user-provided path in the client process."""

    # ``absolute`` normalizes the client-relative location without following a
    # symlink supplied as the snapshot root; import can then reject it.
    return Path(os.path.abspath(Path(raw).expanduser()))


def read_snapshot(raw_path: str) -> tuple[dict[str, Any], Path]:
    """Read and validate a folder or legacy/single-file snapshot."""

    path = local_path(raw_path)
    if path.is_dir():
        if path.is_symlink():
            raise SnapshotIOError(f"snapshot folder must not be a symlink: {path}")
        payload = _read_folder(path)
    elif path.is_file():
        payload = _read_json_object(path, label="snapshot")
    else:
        raise SnapshotIOError(f"snapshot path does not exist: {path}")
    validate_payload(payload)
    return payload, path


def write_snapshot(
    raw_path: str,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a JSON file or a readable snapshot folder."""

    clean = dict(payload)
    validate_payload(clean)
    path = local_path(raw_path)
    if path.exists() and not overwrite:
        raise SnapshotIOError(
            f"export destination already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        _write_json_file_atomic(path, clean, overwrite=overwrite)
    else:
        _write_folder_atomic(path, clean, overwrite=overwrite)
    return path


def validate_payload(payload: Any) -> None:
    """Apply the shared graph/resource snapshot validation contracts."""

    if not isinstance(payload, Mapping):
        raise SnapshotIOError("snapshot JSON must be an object")
    graph = {key: value for key, value in payload.items() if key != "resources"}
    try:
        normalize_snapshot_payload(graph, target_pot_id=str(graph.get("pot_id") or ""))
        if "resources" in payload:
            normalize_resource_snapshot(payload["resources"])
    except (TypeError, ValueError) as exc:
        raise SnapshotIOError(str(exc)) from exc


def _read_folder(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    _reject_symlink_path(manifest_path, root=path)
    if not manifest_path.is_file():
        raise SnapshotIOError(f"snapshot folder is missing {manifest_path.name}")
    manifest = _read_json_object(manifest_path, label="snapshot manifest")
    version = str(manifest.get("format_version", ""))
    if version != "2":
        raise SnapshotIOError(
            f"unsupported folder snapshot format_version {version!r}; expected '2'"
        )
    entities_name = _manifest_filename(manifest, "entities", "entities.json")
    claims_name = _manifest_filename(manifest, "claims", "claims.json")
    entities_path = path / entities_name
    claims_path = path / claims_name
    _reject_symlink_path(entities_path, root=path)
    _reject_symlink_path(claims_path, root=path)
    entities = _read_json_array(entities_path, label="snapshot entities")
    claims = _read_json_array(claims_path, label="snapshot claims")
    payload = {
        "format_version": "2",
        "pot_id": manifest.get("pot_id"),
        "entities": entities,
        "claims": claims,
    }
    if "resources" in manifest:
        resources = manifest["resources"]
        if not isinstance(resources, Mapping):
            raise SnapshotIOError(
                "snapshot manifest field 'resources' must be an object"
            )
        resource_version = str(resources.get("format_version", ""))
        if resource_version != "1":
            raise SnapshotIOError(
                "unsupported resource snapshot format_version "
                f"{resource_version!r}; expected '1'"
            )
        names = resources.get("files")
        if not isinstance(names, list):
            raise SnapshotIOError(
                "snapshot manifest resources field 'files' must be an array"
            )
        normalized_names = [_safe_resource_name(name) for name in names]
        if len(normalized_names) != len(set(normalized_names)):
            raise SnapshotIOError(
                "snapshot manifest lists a resource file more than once"
            )
        files: dict[str, str] = {}
        for name in normalized_names:
            resource_path = path / "resources" / Path(*PurePosixPath(name).parts)
            _reject_symlink_path(resource_path, root=path)
            if not resource_path.is_file():
                raise SnapshotIOError(f"snapshot resource file is missing: {name}")
            try:
                with resource_path.open(
                    "r", encoding="utf-8", newline=""
                ) as resource_file:
                    files[name] = resource_file.read()
            except (OSError, UnicodeDecodeError) as exc:
                raise SnapshotIOError(
                    f"snapshot resource file is not readable UTF-8: {name}"
                ) from exc
        payload["resources"] = {"format_version": "1", "files": files}
    return payload


def _manifest_filename(manifest: Mapping[str, Any], key: str, default: str) -> str:
    value = manifest.get(key, default)
    if not isinstance(value, str) or Path(value).name != value:
        raise SnapshotIOError(f"snapshot manifest field {key!r} must be a file name")
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_json(path, label=label)
    if not isinstance(value, dict):
        raise SnapshotIOError(f"{label} must contain a JSON object: {path}")
    return value


def _read_json_array(path: Path, *, label: str) -> list[Any]:
    value = _read_json(path, label=label)
    if not isinstance(value, list):
        raise SnapshotIOError(f"{label} must contain a JSON array: {path}")
    return value


def _read_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise SnapshotIOError(f"{label} file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise SnapshotIOError(f"{label} is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotIOError(
            f"{label} contains invalid JSON at line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc
    except OSError as exc:
        raise SnapshotIOError(f"could not read {label} {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_file_atomic(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool
) -> None:
    fd, raw_stage = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    stage = Path(raw_stage)
    try:
        _write_json(stage, payload)
        _replace_staged(stage, path, overwrite=overwrite)
    finally:
        stage.unlink(missing_ok=True)


def _write_folder_atomic(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool
) -> None:
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        entities = list(payload.get("entities", []))
        claims = list(payload["claims"])
        manifest: dict[str, Any] = {
            "format_version": "2",
            "pot_id": payload.get("pot_id"),
            "entities": "entities.json",
            "claims": "claims.json",
            "entity_count": len(entities),
            "claim_count": len(claims),
        }
        resources = payload.get("resources")
        if resources is not None:
            normalize_resource_snapshot(resources)
            resource_files = dict(resources["files"])
            manifest["resources"] = {
                "format_version": "1",
                "files": sorted(resource_files),
            }
            for name, contents in resource_files.items():
                target = stage / "resources" / Path(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", encoding="utf-8", newline="") as resource_file:
                    resource_file.write(contents)
        _write_json(stage / "manifest.json", manifest)
        _write_json(stage / "entities.json", entities)
        _write_json(stage / "claims.json", claims)
        (stage / "README.md").write_text(_README, encoding="utf-8")
        _replace_staged(stage, path, overwrite=overwrite)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _replace_staged(stage: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(stage, destination)
        return
    if not overwrite:
        raise SnapshotIOError(
            f"export destination already exists: {destination}; pass --overwrite to replace it"
        )
    backup = destination.with_name(f".{destination.name}.backup-{secrets.token_hex(6)}")
    os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink(missing_ok=True)


def _safe_resource_name(raw_name: Any) -> str:
    if not isinstance(raw_name, str) or not raw_name:
        raise SnapshotIOError("resource snapshot file names must be non-empty strings")
    if "\\" in raw_name or "//" in raw_name or raw_name.endswith("/"):
        raise SnapshotIOError(f"unsafe resource snapshot path: {raw_name!r}")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotIOError(f"unsafe resource snapshot path: {raw_name!r}")
    return path.as_posix()


def _reject_symlink_path(path: Path, *, root: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise SnapshotIOError(f"snapshot folder contains a symlink: {current}")


__all__ = [
    "SnapshotIOError",
    "local_path",
    "read_snapshot",
    "validate_payload",
    "write_snapshot",
]
