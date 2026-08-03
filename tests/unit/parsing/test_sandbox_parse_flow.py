"""Unit tests for the sandbox-based parsing pipeline (Phase 3).

Covers:
* :meth:`CodeGraphService.store_graph_from_artifacts` — the new entry
  point that takes a parsed payload and writes it to neo4j/qdrant.
* :meth:`ParsingService.analyze_workspace` — orchestrator that calls
  ``ProjectSandbox.parse``; Neo4j graph writes are permanently bypassed.
* :meth:`ParsingService.parse_directory` end-to-end with a fake sandbox.

Heavy DB work (real neo4j/qdrant inserts) lives in the
``real_parse``-marked integration tests; these unit tests stub the
sandbox so they run in milliseconds and don't need infra.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.intelligence.tools.sandbox.project_sandbox import (
    ProjectRef,
    ProjectSandbox,
)
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.projects.projects_schema import ProjectStatusEnum
from sandbox import WorkspaceHandle
from sandbox.api.parser_wire import ParseArtifacts


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _node(**overrides) -> SimpleNamespace:
    base = dict(
        id="a.py", node_type="FILE", file="a.py", line=0, end_line=0,
        name="a.py", class_name=None, text=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _edge(**overrides) -> SimpleNamespace:
    base = dict(
        source_id="a.py", target_id="Foo", relationship_type="CONTAINS",
        ident=None, ref_line=None, end_ref_line=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_handle(workspace_id: str = "ws_1") -> WorkspaceHandle:
    return WorkspaceHandle(
        workspace_id=workspace_id,
        branch="main",
        backend_kind="local",
        local_path=f"/tmp/fake/{workspace_id}",
        remote_path=None,
    )


class FakeProjectSandbox:
    """Stand-in for ``ProjectSandbox`` that captures calls and scripts
    return values. Mirrors the real surface only for the methods the
    parsing service touches: ``ensure`` and ``parse``.
    """

    def __init__(self) -> None:
        self.ensure_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        self.handle = _make_handle()
        self.artifacts = ParseArtifacts(
            nodes=[
                _node(),
                _node(id="Foo", node_type="CLASS", line=5, end_line=12,
                      name="Foo", text="class Foo: pass"),
            ],
            relationships=[_edge()],
            repo_dir=".",
            elapsed_s=0.4,
        )
        self.parse_should_raise: Exception | None = None

    async def ensure(self, **kwargs) -> WorkspaceHandle:
        self.ensure_calls.append(kwargs)
        return self.handle

    async def parse(self, handle, **kwargs) -> ParseArtifacts:
        self.parse_calls.append({"handle": handle, **kwargs})
        if self.parse_should_raise is not None:
            raise self.parse_should_raise
        return self.artifacts

    async def health_check(self, _handle) -> bool:
        return True


# ---------------------------------------------------------------------------
# CodeGraphService.store_graph_from_artifacts
# ---------------------------------------------------------------------------


class TestStoreGraphFromArtifacts:
    def test_reconstructs_and_writes(self, monkeypatch):
        """The new entry point should reconstruct an nx graph from
        artifacts and feed it into the same _store_graph DB-write path
        the legacy create_and_store_graph already uses. We check the
        seam between the two halves rather than the DB writes
        themselves (covered by integration tests)."""
        from app.modules.parsing.graph_construction import code_graph_service

        # Stub the heavy reconstruction import so we don't drag in
        # tree_sitter just to check the wiring.
        fake_graph = MagicMock(name="nx.MultiDiGraph")
        fake_graph.number_of_nodes.return_value = 7
        fake_graph.number_of_edges.return_value = 3
        with patch.object(
            code_graph_service,
            "_reconstruct_graph_from_payload",
            return_value=fake_graph,
            create=True,
        ):
            # parsing_repomap is imported lazily inside the method;
            # patch the module attribute the import resolves to.
            from app.modules.parsing.graph_construction import parsing_repomap

            with patch.object(
                parsing_repomap,
                "_reconstruct_graph_from_payload",
                return_value=fake_graph,
            ):
                service = code_graph_service.CodeGraphService.__new__(
                    code_graph_service.CodeGraphService
                )
                service._store_graph = MagicMock()  # type: ignore[attr-defined]

                artifacts = ParseArtifacts(
                    nodes=[_node()], relationships=[_edge()]
                )
                service.store_graph_from_artifacts(
                    artifacts, project_id="p1", user_id="u1"
                )
                service._store_graph.assert_called_once_with(
                    fake_graph, "p1", "u1"
                )


# ---------------------------------------------------------------------------
# ParsingService.analyze_workspace
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sandbox() -> FakeProjectSandbox:
    return FakeProjectSandbox()


@pytest.fixture
def parsing_service(fake_sandbox):
    """Construct a ParsingService with mocked DB / project services so we
    can exercise orchestration without standing up Postgres or Neo4j.
    InferenceService is no longer constructed (Neo4j bypassed).
    """
    from app.modules.parsing.graph_construction import parsing_service as ps_mod

    db = MagicMock(name="db_session")
    with patch.object(ps_mod, "ProjectService") as ProjectServiceMock, \
         patch.object(ps_mod, "SearchService"), \
         patch.object(ps_mod, "CodeProviderService"), \
         patch.object(ps_mod, "ParseHelper"):
        project_service = ProjectServiceMock.return_value
        project_service.get_project_from_db_by_id = AsyncMock(
            return_value={"project_name": "owner/repo", "branch_name": "main"}
        )
        project_service.update_project_status = AsyncMock()

        service = ps_mod.ParsingService(
            db=db,
            user_id="u1",
            neo4j_config={"uri": "bolt://x", "username": "u", "password": "p"},
            raise_library_exceptions=True,
            project_sandbox=fake_sandbox,  # type: ignore[arg-type]
        )
        service.project_service = project_service
        yield service


@pytest.mark.asyncio
async def test_analyze_workspace_happy_path(
    parsing_service, fake_sandbox
):
    handle = _make_handle()
    repo_details = ParsingRequest(
        repo_name="owner/repo", branch_name="main",
        repo_path="https://github.com/owner/repo.git", commit_id="deadbeef",
    )

    await parsing_service.analyze_workspace(
        handle=handle, project_id="p1", user_id="u1",
        user_email="x@y.com", repo_details=repo_details,
    )

    # Neo4j permanently bypassed: parse runs, graph/inference do not.
    assert len(fake_sandbox.parse_calls) == 1
    assert fake_sandbox.parse_calls[0]["handle"] is handle

    statuses = [
        call.args[1]
        for call in parsing_service.project_service.update_project_status.await_args_list
    ]
    assert ProjectStatusEnum.READY in statuses
    assert ProjectStatusEnum.ERROR not in statuses


@pytest.mark.asyncio
async def test_analyze_workspace_file_only_still_ready_when_neo4j_bypassed(
    parsing_service, fake_sandbox
):
    """FILE-only repos (docs/HTML/arrow-only JS) used to trip the old
    language gate. With Neo4j bypassed they must still become READY."""
    fake_sandbox.artifacts = ParseArtifacts(
        nodes=[_node(), _node(id="b.py", file="b.py", name="b.py")],
        relationships=[],
    )
    handle = _make_handle()
    repo_details = ParsingRequest(
        repo_name="owner/docs", branch_name="main",
        repo_path=None, commit_id="abc",
    )

    await parsing_service.analyze_workspace(
        handle=handle, project_id="p1", user_id="u1",
        user_email="", repo_details=repo_details,
    )

    statuses = [
        call.args[1]
        for call in parsing_service.project_service.update_project_status.await_args_list
    ]
    assert ProjectStatusEnum.READY in statuses


@pytest.mark.asyncio
async def test_analyze_workspace_rejects_empty_artifact_stream(
    parsing_service, fake_sandbox
):
    """Truly empty parse output still fails — nothing was cloned/indexed."""
    fake_sandbox.artifacts = ParseArtifacts(nodes=[], relationships=[])
    handle = _make_handle()
    repo_details = ParsingRequest(
        repo_name="owner/empty", branch_name="main",
        repo_path=None, commit_id="abc",
    )

    from app.modules.parsing.graph_construction.parsing_helper import (
        ParsingFailedError,
    )

    with pytest.raises(
        ParsingFailedError, match=r"language currently supported"
    ):
        await parsing_service.analyze_workspace(
            handle=handle, project_id="p1", user_id="u1",
            user_email="", repo_details=repo_details,
        )


@pytest.mark.asyncio
async def test_analyze_workspace_propagates_parser_failure(
    parsing_service, fake_sandbox
):
    """A parser-internal failure (e.g. potpie-parse exited non-zero)
    should bubble up as ParsingServiceError when the service is in
    library mode, with no graph write or inference run attempted."""
    fake_sandbox.parse_should_raise = RuntimeError("potpie-parse exited 1")
    handle = _make_handle()
    repo_details = ParsingRequest(
        repo_name="owner/repo", branch_name="main",
        repo_path=None, commit_id="abc",
    )

    from app.modules.parsing.graph_construction.parsing_helper import (
        ParsingServiceError,
    )

    with pytest.raises(ParsingServiceError, match="Sandbox parse failed"):
        await parsing_service.analyze_workspace(
            handle=handle, project_id="p1", user_id="u1",
            user_email="", repo_details=repo_details,
        )


# ---------------------------------------------------------------------------
# parse_directory end-to-end (with the sandbox flow stubbed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_directory_uses_sandbox_path(
    parsing_service, fake_sandbox
):
    """Top-level smoke: parse_directory should provision the sandbox,
    parse via it, and persist the result. No call to clone_or_copy /
    setup_project_directory / detect_repo_language anywhere."""
    from app.modules.parsing.graph_construction import parsing_service as ps_mod

    # The project exists in the DB but is in a non-INFERRING state, so
    # we bypass the early-return paths and exercise the main flow.
    # analyze_workspace calls get_project_from_db_by_id again to fetch
    # repo_name/branch_name for the post-parse email; it sees the same dict.
    parsing_service.project_service.get_project_from_db_by_id = AsyncMock(
        return_value={
            "project_name": "owner/repo",
            "branch_name": "main",
            "status": ProjectStatusEnum.SUBMITTED.value,
        }
    )
    parsing_service.project_service.update_project_status = AsyncMock()

    repo_details = ParsingRequest(
        repo_name="owner/repo",
        branch_name="main",
        repo_path="https://github.com/owner/repo.git",
        commit_id="deadbeef",
    )

    # Stub GithubService inside the parsing_service module since
    # _resolve_user_github_token instantiates it directly.
    with patch.object(ps_mod, "GithubService") as GithubServiceMock:
        GithubServiceMock.return_value.get_github_oauth_token.return_value = "tk"
        out = await parsing_service.parse_directory(
            repo_details=repo_details,
            user_id="u1",
            user_email="x@y.com",
            project_id="p1",
            cleanup_graph=False,  # skip the cleanup call to keep this focused
        )

    assert out["id"] == "p1"
    assert "successfully" in out["message"].lower()

    # Sandbox was provisioned with ANALYSIS-friendly parameters.
    assert len(fake_sandbox.ensure_calls) == 1
    ensure_kwargs = fake_sandbox.ensure_calls[0]
    assert ensure_kwargs["user_id"] == "u1"
    assert ensure_kwargs["project_id"] == "p1"
    assert isinstance(ensure_kwargs["repo"], ProjectRef)
    assert ensure_kwargs["repo"].repo_name == "owner/repo"
    assert ensure_kwargs["repo"].base_ref == "deadbeef"
    assert ensure_kwargs["auth_token"] == "tk"

    # Parse happened; Neo4j graph write is permanently bypassed.
    assert len(fake_sandbox.parse_calls) == 1
