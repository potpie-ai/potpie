"""Who may read commit history and roll back, and who the receipts name.

Commit access on a local install is granted per typed operation, for exactly
the pot the resource manager selected; nothing else in the process gets it by
default. Receipts and previews name the local owner and a digest of the home,
never the account.
"""

from __future__ import annotations

import asyncio
import getpass
from pathlib import Path

import pytest

from potpie.runtime.commit_access import (
    LOCAL_COMMIT_ACTOR,
    authorize_local_commit,
    commit_grant,
    local_commit_actor,
    local_commit_host,
)
from potpie.runtime.composition import build_local_runtime
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.commit_service import CommitAccessDenied

pytestmark = pytest.mark.unit


def _authorize(pot_id: str, access: str) -> None:
    asyncio.run(authorize_local_commit(pot_id, access))


def test_no_grant_means_no_commit_access() -> None:
    for access in ("read", "write", "admin"):
        with pytest.raises(CommitAccessDenied):
            _authorize("pot_1", access)


def test_a_grant_covers_only_its_own_pot_and_levels() -> None:
    with commit_grant("pot_1"):
        for access in ("read", "write", "admin"):
            _authorize("pot_1", access)
        with pytest.raises(CommitAccessDenied, match="pot_1"):
            _authorize("pot_2", "read")

    with commit_grant("pot_1", {"read"}):
        _authorize("pot_1", "read")
        with pytest.raises(CommitAccessDenied, match="write"):
            _authorize("pot_1", "write")

    with pytest.raises(CommitAccessDenied):
        _authorize("pot_1", "read")


def test_unknown_access_levels_are_rejected() -> None:
    with pytest.raises(ValueError):
        _authorize("pot_1", "owner")
    with pytest.raises(ValueError):
        with commit_grant("pot_1", {"owner"}):
            pass


def test_the_local_actor_and_host_never_name_the_account(tmp_path: Path) -> None:
    host = local_commit_host(tmp_path)

    assert local_commit_actor() == LOCAL_COMMIT_ACTOR == "local:owner"
    assert getpass.getuser() not in LOCAL_COMMIT_ACTOR
    assert host.startswith("local:") and str(tmp_path) not in host
    assert getpass.getuser() not in host
    assert local_commit_host(tmp_path) == host
    assert local_commit_host(tmp_path / "other") != host


def test_the_composed_runtime_passes_explicit_commit_wiring(tmp_path: Path) -> None:
    runtime = build_local_runtime(backend=InMemoryGraphBackend())
    commits = runtime.engine.graph_workbench.commit_service

    assert commits.authorize is authorize_local_commit
    assert commits.actor() == LOCAL_COMMIT_ACTOR
    assert commits.mirror is not None and commits.previews is not None
    assert Path(commits.mirror.path).name == "graph_commits.sqlite"
    assert Path(commits.previews.path).name == "rollback_previews.sqlite"
    assert getpass.getuser() not in commits.host
    # Document imports and removals are journaled resource workflows.
    assert runtime.engine.resources.journal is not None
    # Composed but not granted: the runtime itself refuses to serve history.
    with pytest.raises(CommitAccessDenied):
        asyncio.run(commits.journal_status_async(pot_id="pot_1"))
