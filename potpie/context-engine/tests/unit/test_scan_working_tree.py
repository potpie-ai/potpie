"""ScanWorkingTreeUseCase wires scanners → canonical writer (P4)."""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

import pytest

from adapters.outbound.scanners.codeowners import CodeownersScanner
from adapters.outbound.scanners.kubernetes_manifest import KubernetesManifestScanner
from application.services.config_scanner_registry import ConfigSourceScannerRegistry
from application.use_cases.scan_working_tree import (
    ScanWorkingTreeUseCase,
    WorkingTreeFile,
)
from domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef


class _FakeCanonicalWriter:
    def __init__(self) -> None:
        self.entity_calls: list[tuple[str, list[EntityUpsert], ProvenanceRef]] = []
        self.edge_calls: list[tuple[str, list[EdgeUpsert], ProvenanceRef]] = []

    async def upsert_entities(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        self.entity_calls.append((pot_id, items, provenance))
        return len(items)

    async def upsert_edges(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        self.edge_calls.append((pot_id, items, provenance))
        return len(items)


def _build_registry() -> ConfigSourceScannerRegistry:
    registry = ConfigSourceScannerRegistry()
    registry.register(CodeownersScanner())
    registry.register(KubernetesManifestScanner())
    return registry


MANIFEST = dedent(
    """
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: auth-svc
      namespace: auth
      labels:
        app.kubernetes.io/name: auth-svc
    """
).strip()


class TestExecuteEndToEnd:
    async def test_emits_entities_and_edges_through_writer(self) -> None:
        writer = _FakeCanonicalWriter()
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=_build_registry(),
            canonical_writer=writer,
        )
        result = await use_case.execute(
            pot_id="pot-1",
            run_id="run-abc",
            working_tree=[
                WorkingTreeFile(
                    path="apps/auth/CODEOWNERS",
                    content="*  @alice\n",
                    repo_name="acme/api",
                    commit_sha="aaa",
                ),
                WorkingTreeFile(
                    path="clusters/prod/auth-svc.yaml",
                    content=MANIFEST,
                    repo_name="acme/api",
                    commit_sha="aaa",
                ),
            ],
            observed_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        )
        assert result.entities_upserted > 0
        assert result.edges_upserted > 0
        assert set(result.scanners_run) >= {"codeowners", "kubernetes-manifest"}
        assert writer.entity_calls and writer.edge_calls
        # Provenance threaded through
        prov = writer.edge_calls[0][2]
        assert prov.pot_id == "pot-1"
        assert prov.source_event_id == "run-abc"
        assert prov.source_system == "config-scanner"

    async def test_empty_working_tree_writes_nothing(self) -> None:
        writer = _FakeCanonicalWriter()
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=_build_registry(),
            canonical_writer=writer,
        )
        result = await use_case.execute(
            pot_id="pot-1",
            run_id="run-abc",
            working_tree=[],
        )
        assert result.entities_upserted == 0
        assert result.edges_upserted == 0
        assert writer.entity_calls == [] and writer.edge_calls == []

    async def test_no_matching_scanner_short_circuits(self) -> None:
        writer = _FakeCanonicalWriter()
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=_build_registry(),
            canonical_writer=writer,
        )
        result = await use_case.execute(
            pot_id="pot-1",
            run_id="run-abc",
            working_tree=[
                WorkingTreeFile(path="README.md", content="# hello"),
            ],
        )
        assert result.entities_upserted == 0
        assert result.edges_upserted == 0
        assert writer.entity_calls == [] and writer.edge_calls == []

    async def test_missing_pot_id_raises(self) -> None:
        writer = _FakeCanonicalWriter()
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=_build_registry(),
            canonical_writer=writer,
        )
        with pytest.raises(ValueError):
            await use_case.execute(pot_id="", run_id="r", working_tree=[])
        with pytest.raises(ValueError):
            await use_case.execute(pot_id="p", run_id="", working_tree=[])

    async def test_warnings_propagated(self) -> None:
        writer = _FakeCanonicalWriter()
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=_build_registry(),
            canonical_writer=writer,
        )
        # Malformed manifest → YAML parse warning bubbles up
        result = await use_case.execute(
            pot_id="pot-1",
            run_id="r",
            working_tree=[
                WorkingTreeFile(
                    path="clusters/prod/broken.yaml",
                    content="::: not yaml :::",
                    repo_name="acme/api",
                )
            ],
        )
        assert any("YAML parse error" in w for w in result.warnings)


class TestTranslation:
    async def test_edge_carries_evidence_strength_deterministic(self) -> None:
        writer = _FakeCanonicalWriter()
        use_case = ScanWorkingTreeUseCase(
            scanner_registry=_build_registry(),
            canonical_writer=writer,
        )
        await use_case.execute(
            pot_id="pot-1",
            run_id="r",
            working_tree=[
                WorkingTreeFile(
                    path="apps/auth/CODEOWNERS",
                    content="*  @alice\n",
                    repo_name="acme/api",
                )
            ],
        )
        edges = writer.edge_calls[0][1]
        assert all(
            e.properties.get("evidence_strength") == "deterministic" for e in edges
        )
        assert all("fact" in e.properties for e in edges)
        assert all("source_ref" in e.properties for e in edges)
