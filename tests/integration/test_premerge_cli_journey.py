"""Pre-merge journey test: fresh clone -> install -> setup -> graph mutation.

This test validates the critical first-user path inside an isolated temp
environment:
1) clone the repository into a brand-new directory,
2) install dependencies from lockfiles,
3) run non-interactive setup,
4) create graph data through ``potpie graph mutate``,
5) verify the graph can be queried.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.premerge_journey]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTEXT_ENGINE_PROJECT = "potpie/context-engine"


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    step: str,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{step} failed\n"
        f"command: {' '.join(cmd)}\n"
        f"exit_code: {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    return proc


def _run_json_cli(
    clone_root: Path,
    env: dict[str, str],
    *args: str,
    timeout: int = 300,
) -> dict[str, Any]:
    proc = _run(
        ["uv", "run", "--project", _CONTEXT_ENGINE_PROJECT, "potpie", "--json", *args],
        cwd=clone_root,
        env=env,
        timeout=timeout,
        step=f"potpie {' '.join(args)}",
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - assertion helper
        raise AssertionError(
            f"Expected JSON output for 'potpie {' '.join(args)}' but got:\n{proc.stdout}"
        ) from exc


def _unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return payload


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    xdg = tmp_path / "xdg"
    home = tmp_path / "home"
    xdg.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    source_home = Path(os.environ.get("HOME", Path.home()))
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(xdg)
    env["HOME"] = str(home)
    env["CONTEXT_ENGINE_HOST_MODE"] = "in_process"
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    # Keep rustup/cargo on the host toolchain when HOME is redirected for Potpie
    # config isolation — otherwise `cargo` exists on PATH but has no toolchain.
    env.setdefault(
        "RUSTUP_HOME", os.environ.get("RUSTUP_HOME", str(source_home / ".rustup"))
    )
    env.setdefault(
        "CARGO_HOME", os.environ.get("CARGO_HOME", str(source_home / ".cargo"))
    )
    return env


def _has_usable_rust_toolchain(env: dict[str, str]) -> bool:
    try:
        proc = subprocess.run(
            ["cargo", "--version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def test_premerge_journey_from_fresh_clone_creates_context_graph(
    tmp_path: Path,
) -> None:
    clone_root = tmp_path / "repo-clone"
    env = _isolated_env(tmp_path)
    if not _has_usable_rust_toolchain(env):
        message = "Rust toolchain (cargo) is required for fresh-clone workspace sync"
        if os.environ.get("PREMERGE_JOURNEY_REQUIRED") == "1":
            pytest.fail(message)
        pytest.skip(message)

    _run(
        ["git", "clone", "--depth", "1", str(_REPO_ROOT), str(clone_root)],
        cwd=tmp_path,
        env=env,
        timeout=60,
        step="git clone",
    )

    _run(
        ["uv", "sync", "--frozen", "--all-packages", "--no-cache"],
        cwd=clone_root,
        env=env,
        timeout=240,
        step="uv sync",
    )

    setup_payload = _run_json_cli(
        clone_root,
        env,
        "setup",
        "--repo",
        ".",
        "--agent",
        "claude",
        "--backend",
        "embedded",
        "--yes",
        "--in-process",
        timeout=120,
    )
    setup = _unwrap_result(setup_payload)
    assert setup.get("ok") is True
    step_states = {
        row.get("step"): row.get("state")
        for row in setup.get("steps", [])
        if isinstance(row, dict)
    }
    assert step_states.get("source") in {"done", "skipped"}

    source_payload = _run_json_cli(
        clone_root, env, "source", "add", "repo", ".", timeout=60
    )
    source_add = _unwrap_result(source_payload)
    assert source_add.get("kind") == "repo"
    assert source_add.get("registration_only") is True

    mutation_file = tmp_path / "journey-mutation.json"
    mutation_file.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {
                            "key": "service:journey-service",
                            "type": "Service",
                            "name": "journey-service",
                            "summary": "Service used by the pre-merge journey test.",
                            "description": "Synthetic service entity for end-to-end CI validation.",
                        },
                    },
                    {
                        "op": "upsert_entity",
                        "subject": {
                            "key": "service:journey-ledger",
                            "type": "Service",
                            "name": "journey-ledger",
                            "summary": "Dependency used by the pre-merge journey test.",
                            "description": "Synthetic dependency entity for end-to-end CI validation.",
                        },
                    },
                    {
                        "op": "link_entities",
                        "subgraph": "infra_topology",
                        "subject": {
                            "key": "service:journey-service",
                            "type": "Service",
                        },
                        "predicate": "DEPENDS_ON",
                        "object": {"key": "service:journey-ledger", "type": "Service"},
                        "truth": "source_observation",
                        "description": "journey service depends on journey ledger",
                        "evidence": [
                            {
                                "source_ref": "test:premerge-journey",
                                "authority": "repository_metadata",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    mutate_payload = _run_json_cli(
        clone_root,
        env,
        "graph",
        "mutate",
        "--file",
        str(mutation_file),
        timeout=60,
    )
    mutate = _unwrap_result(mutate_payload)
    assert mutate.get("status") in {"applied", "committed"}
    diff_counts = mutate.get("diff") or {}
    assert int(diff_counts.get("edge_upserts", 0)) >= 1

    read_payload = _run_json_cli(
        clone_root,
        env,
        "graph",
        "read",
        "--subgraph",
        "infra_topology",
        "--view",
        "service_neighborhood",
        "--scope",
        "service:journey-service",
        "--limit",
        "10",
        timeout=60,
    )
    read_result = _unwrap_result(read_payload)
    assert read_result.get("items"), (
        "expected graph read to return journey topology items"
    )

    status_payload = _run_json_cli(clone_root, env, "graph", "status", timeout=60)
    status = _unwrap_result(status_payload)
    claim_count = int(
        (((status.get("backend") or {}).get("counts") or {}).get("claims") or 0)
    )
    assert claim_count >= 1
