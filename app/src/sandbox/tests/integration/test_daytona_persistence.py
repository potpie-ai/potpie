"""End-to-end persistence test against a real Daytona stack.

Reproduces the bug the user hit in the agent: previously, the second message
of a conversation came back to a fresh worker, the in-memory
``_project_sandbox_ids`` map was empty, and the provider spun up a brand-new
sandbox — orphaning the agent's prior commits in the original sandbox. With
the label-based recovery path (see ``DaytonaWorkspaceProvider._recover_project_sandbox``)
the second worker adopts the existing sandbox and the work survives.

The test simulates a worker restart by building two independent
``DaytonaWorkspaceProvider`` instances against the same Daytona client:

    1. provider_a: clone, write a marker file inside the worktree
    2. provider_b: same identity, read the marker → must succeed

If either step fails, the bug has regressed.

Skipped when ``DAYTONA_API_URL``/``DAYTONA_API_KEY`` aren't set so the test
suite stays runnable in environments without the Daytona stack. To run::

    export $(grep -v '^#' app/src/sandbox/.env.daytona.local | xargs)
    uv run --extra dev --extra daytona pytest \\
        app/src/sandbox/tests/integration/test_daytona_persistence.py -v

The fixture does its own cleanup in a ``finally`` block — the test deletes
the sandbox via the SDK whether or not it passed.
"""

from __future__ import annotations

import os
import secrets
import shlex
from typing import Any

import pytest

from sandbox.adapters.outbound.daytona.provider import (
    DaytonaRuntimeProvider,
    DaytonaWorkspaceProvider,
)
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)


def _daytona_available() -> bool:
    return bool(os.getenv("DAYTONA_API_URL") and os.getenv("DAYTONA_API_KEY"))


pytestmark = pytest.mark.skipif(
    not _daytona_available(),
    reason="Requires a running Daytona stack — see .env.daytona.local.",
)


def _build_service(client: Any) -> tuple[DaytonaWorkspaceProvider, SandboxService]:
    """Two providers share the SDK client but each has its own in-memory state.

    That's the worker-restart simulation: same Daytona, fresh
    ``_project_sandbox_ids`` map.
    """
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        snapshot=os.getenv("DAYTONA_SNAPSHOT", "potpie/agent-sandbox:0.1.0"),
        workspace_root=os.getenv("DAYTONA_WORKSPACE_ROOT", "/home/agent/work"),
    )
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=DaytonaRuntimeProvider(workspace_provider),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )
    return workspace_provider, service


@pytest.mark.asyncio
async def test_changes_persist_across_worker_restart() -> None:
    """A marker written by provider_a must be visible to a fresh provider_b
    using the same conversation_id."""
    from daytona import Daytona

    client = Daytona()

    # Unique-per-test identity. Worker restart simulation == new providers
    # against the same identity. We delete the sandbox at the end either way.
    user_id = "test-persist"
    project_id = f"p-{secrets.token_hex(4)}"
    conversation_id = f"conv-{secrets.token_hex(4)}"
    repo = RepoIdentity(
        repo_name="octocat/Hello-World",
        repo_url="https://github.com/octocat/Hello-World.git",
    )
    request = WorkspaceRequest(
        user_id=user_id,
        project_id=project_id,
        repo=repo,
        base_ref="master",
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
        create_branch=True,
    )

    marker = f"persist-{secrets.token_hex(4)}"
    provider_a, service_a = _build_service(client)
    provider_b: DaytonaWorkspaceProvider | None = None
    sandbox_id: str | None = None
    try:
        # --- worker A: clone + write a marker file inside the worktree ---
        ws_a = await service_a.get_or_create_workspace(request)
        sandbox_id = ws_a.location.backend_workspace_id
        assert sandbox_id, "Daytona didn't return a sandbox id"
        await service_a.get_or_create_runtime(RuntimeRequest(ws_a.id))
        write_result = await service_a.exec(
            ws_a.id,
            ExecRequest(
                cmd=("bash", "-c", f"echo {shlex.quote(marker)} > marker.txt"),
            ),
        )
        assert write_result.exit_code == 0, write_result.stdout

        # --- worker restart: new provider, same Daytona client. Recovery
        # must adopt the existing sandbox via labels, not create a new one. ---
        provider_b, service_b = _build_service(client)
        ws_b = await service_b.get_or_create_workspace(request)
        assert ws_b.location.backend_workspace_id == sandbox_id, (
            f"recovery failed: got fresh sandbox {ws_b.location.backend_workspace_id} "
            f"instead of adopting {sandbox_id}"
        )
        assert ws_b.location.remote_path == ws_a.location.remote_path
        await service_b.get_or_create_runtime(RuntimeRequest(ws_b.id))
        read_result = await service_b.exec(
            ws_b.id,
            ExecRequest(cmd=("cat", "marker.txt")),
        )
        assert read_result.exit_code == 0
        assert read_result.stdout.strip() == marker.encode(), (
            f"marker round-trip failed: wrote {marker!r}, "
            f"read {read_result.stdout!r}"
        )
    finally:
        # Always clean up the sandbox so repeated runs don't accumulate.
        if sandbox_id:
            sandbox = (
                provider_a._sandboxes.get(sandbox_id)
                or (provider_b._sandboxes.get(sandbox_id) if provider_b else None)
            )
            if sandbox is None:
                try:
                    sandbox = client.get(sandbox_id)
                except Exception:
                    sandbox = None
            if sandbox is not None and hasattr(sandbox, "delete"):
                try:
                    sandbox.delete()
                except Exception:
                    pass
