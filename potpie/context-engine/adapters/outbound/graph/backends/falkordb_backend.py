"""FalkorDB ``GraphBackend`` profile.

This backend wraps the existing FalkorDB reader/writer adapters behind the same
capability bundle used by Neo4j. Application services see only
``GraphBackend``; Falkor-specific graph handles, Lite/server mode, and Cypher
shim details stay in outbound adapters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from adapters.outbound.graph.apply_plan import apply_mutation_batch
from adapters.outbound.graph.backends._unimplemented import (
    UnimplementedSnapshot,
)
from adapters.outbound.graph.backends.claim_query_analytics import ClaimQueryAnalytics
from adapters.outbound.graph.backends.claim_query_semantic import (
    ClaimQuerySemanticSearch,
)
from adapters.outbound.graph.falkordb_inspection import FalkorDBInspection
from adapters.outbound.graph.falkordb_reader import FalkorDBClaimQueryStore
from adapters.outbound.graph.falkordb_writer import (
    FalkorDBGraphProvider,
    FalkorDBGraphWriter,
    _records_from_result,
)
from adapters.outbound.graph.canonical_claim_query import (
    card_for_row,
    row_from_record,
)
from adapters.outbound.graph.entity_summary_repair import (
    ENTITY_SUMMARY_REPAIR_LIMIT,
    ENTITY_SUMMARY_SCAN_CYPHER,
    ENTITY_SUMMARY_UPDATE_CYPHER,
    repaired_entity_properties,
)
from adapters.outbound.graph.semantic_index_repair import (
    SEMANTIC_INDEX_REPAIR_LIMIT,
    claim_needs_reembed,
    stored_dim_mismatch,
)
from adapters.outbound.graph.writer_port import GraphWriterPort
from domain.errors import CapabilityNotImplemented
from domain.graph_mutations import ProvenanceContext
from domain.lifecycle import DONE, FAILED, SetupPlan, StepResult
from domain.ports.claim_query import ClaimQueryPort
from domain.ports.graph.backend import BackendCapabilities
from domain.ports.graph.mutation import BackendReadiness
from domain.reconciliation import MutationBatch, MutationResult

logger = logging.getLogger(__name__)

_PROFILE = "falkordb"
_LITE_PROFILE = "falkordb_lite"

# All repair Cypher is edge-first with UNBOUND endpoints — anchoring a node
# is the embedded-FalkorDB plan shape that returns zero rows when the bound
# node is internal id 0 (see FIND_CLAIMS_CYPHER).
_SEMANTIC_SCAN_CYPHER = """
MATCH ()-[r:RELATES_TO {group_id: $gid}]->()
RETURN r{.*} AS props
LIMIT $limit
"""

_REEMBED_BY_CLAIM_KEY_CYPHER = """
MATCH ()-[r:RELATES_TO {group_id: $gid, claim_key: $claim_key}]->()
SET r.fact_embedding = vecf32($embedding),
    r.embedding_model = $embedding_model,
    r.embedding_dim = $embedding_dim
RETURN count(r) AS updated
"""

# ``source_ref`` is part of the edge identity for FalkorDB writes: a null
# param must match only edges that themselves have no source_ref, never act
# as a wildcard across every edge sharing the (predicate, subject, object)
# tuple.
_REEMBED_BY_TUPLE_CYPHER = """
MATCH ()-[r:RELATES_TO {group_id: $gid, name: $predicate, subject_key: $subject_key, object_key: $object_key}]->()
WHERE (($source_ref IS NULL AND r.source_ref IS NULL) OR r.source_ref = $source_ref)
SET r.fact_embedding = vecf32($embedding),
    r.embedding_model = $embedding_model,
    r.embedding_dim = $embedding_dim
RETURN count(r) AS updated
"""

_DROP_VECTOR_INDEX_CYPHER = (
    "DROP VECTOR INDEX FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)"
)


class _FalkorDBModeSettings:
    """Settings adapter that pins FalkorDB runtime mode for a backend profile."""

    def __init__(self, base: Any, mode: str) -> None:
        self._base = base
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def falkordb_mode(self) -> str:
        return self._mode


def _run_sync(coro: Any) -> Any:
    """Drive an async writer call from a sync backend port."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "FalkorDBGraphBackend sync mutation cannot run inside an event loop; "
        "use the async door (mutation.apply_async)."
    )


@dataclass(slots=True)
class _FalkorDBMutation:
    settings: Any
    writer: GraphWriterPort
    profile: str = _PROFILE

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        return await apply_mutation_batch(
            self.writer,
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        return _run_sync(
            self.apply_async(
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
            )
        )

    def invalidate(
        self, *, pot_id: str, claim_keys: Any, reason: str | None = None
    ) -> int:
        raise CapabilityNotImplemented(
            f"graph.{self.profile}.mutation.invalidate",
            detail=f"claim-key invalidation is not implemented for {self.profile} yet",
            recommended_next_action="use mutation.apply with InvalidationOp, or implement claim-key Cypher invalidation",
        )

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        return _run_sync(self.writer.reset_pot(pot_id))

    def readiness(self, pot_id: str) -> BackendReadiness:
        ready = bool(getattr(self.writer, "enabled", False))
        return BackendReadiness(
            profile=self.profile,
            ready=ready,
            detail=(
                f"{self.profile} claim_query + mutation + semantic + analytics + "
                "inspection wired; snapshot pending"
                if ready
                else f"{self.profile} backend is not configured or context graph is disabled"
            ),
            capability_ready={
                "mutation": ready,
                "claim_query": ready,
                "analytics": ready,
                "semantic": ready,
                "inspection": ready,
                "snapshot": False,
            },
        )


@dataclass(slots=True)
class FalkorDBGraphBackend:
    """FalkorDB-backed ``GraphBackend``."""

    settings: Any
    writer: GraphWriterPort | None = None
    graph_provider: FalkorDBGraphProvider | None = None
    embedder: Any = None
    profile_name: str = _PROFILE
    force_mode: str | None = None
    _claim_query: ClaimQueryPort = field(init=False)
    _mutation: _FalkorDBMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)

    def __post_init__(self) -> None:
        if self.force_mode is not None:
            self.settings = _FalkorDBModeSettings(self.settings, self.force_mode)
        provider = self.graph_provider or FalkorDBGraphProvider(self.settings)
        writer = self.writer or FalkorDBGraphWriter(
            self.settings, graph_provider=provider, embedder=self.embedder
        )
        self.graph_provider = provider
        self.writer = writer
        self._claim_query = FalkorDBClaimQueryStore(
            self.settings, graph_provider=provider, embedder=self.embedder
        )
        self._mutation = _FalkorDBMutation(
            self.settings, writer, profile=self.profile_name
        )
        self._semantic = ClaimQuerySemanticSearch(self._claim_query)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.writer, "enabled", False))

    @property
    def profile(self) -> str:
        return self.profile_name

    @property
    def graph_writer(self) -> GraphWriterPort:
        """Compatibility alias for old ingestion paths that seed via writer."""
        assert self.writer is not None
        return self.writer

    @property
    def claim_query(self) -> ClaimQueryPort:
        return self._claim_query

    @property
    def mutation(self) -> _FalkorDBMutation:
        return self._mutation

    @property
    def semantic(self) -> ClaimQuerySemanticSearch:
        return self._semantic

    @property
    def inspection(self) -> FalkorDBInspection:
        return FalkorDBInspection(
            self.settings,
            graph_provider=self.graph_provider,
            embedder=self.embedder,
        )

    @property
    def analytics(self) -> ClaimQueryAnalytics:
        return ClaimQueryAnalytics(
            self._claim_query,
            entity_summary_repair=self._repair_entity_summaries,
            semantic_index_repair=self._repair_semantic_index,
        )

    @property
    def snapshot(self) -> UnimplementedSnapshot:
        return UnimplementedSnapshot(self.profile_name)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=self.profile_name,
            mutation=True,
            claim_query=True,
            analytics=True,
            semantic=True,
            inspection=True,
            snapshot=False,
        )

    def provision(self, plan: SetupPlan) -> StepResult:
        if not self.enabled:
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=f"{self.profile_name} backend is not configured or context graph is disabled",
                metadata={"profile": self.profile_name},
            )
        try:
            ok = bool(_run_sync(self.graph_writer.ensure_indexes()))
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=str(exc),
                metadata={"profile": self.profile_name},
            )
        return StepResult(
            step="backend.provision",
            state=DONE if ok else FAILED,
            detail=(
                f"{self.profile_name} backend ready"
                if ok
                else f"{self.profile_name} index setup failed"
            ),
            metadata={"profile": self.profile_name},
        )

    def _repair_semantic_index(self, pot_id: str) -> dict[str, Any]:
        """Re-embed claims whose embeddings are missing, stale, or mis-sized.

        This is both the recovery pass for failed embedding attaches and the
        migration pass for embedder changes (model or dimensions). On a
        dimension change the vector index is dropped first — re-embedded
        values at the new size must not stream into an index built for the
        old one — and recreated after the claims are rewritten.
        """
        if self.embedder is None:
            return {
                "scanned": 0,
                "repaired": 0,
                "failed": 0,
                "detail": "no embedder configured; nothing to re-embed",
            }
        embedder_name = str(getattr(self.embedder, "name", "unknown"))
        embedder_dim = int(getattr(self.embedder, "dimensions", 0))
        graph = self.graph_provider()
        rows = _records_from_result(
            graph.query(
                _SEMANTIC_SCAN_CYPHER,
                params={"gid": pot_id, "limit": SEMANTIC_INDEX_REPAIR_LIMIT},
            )
        )
        props_list = [
            props for rec in rows if isinstance(props := rec.get("props"), Mapping)
        ]
        needs = [
            props
            for props in props_list
            if claim_needs_reembed(
                props, embedder_name=embedder_name, embedder_dim=embedder_dim
            )
        ]
        index_recreated = False
        index_errors: list[str] = []
        if any(
            stored_dim_mismatch(props, embedder_dim=embedder_dim) for props in needs
        ):
            try:
                graph.query(_DROP_VECTOR_INDEX_CYPHER)
                index_recreated = True
            except Exception as exc:  # noqa: BLE001
                index_errors.append(f"index drop failed: {exc}")
                logger.warning("vector index drop failed (continuing): %s", exc)
        repaired = 0
        failed = 0
        for props in needs:
            try:
                row = row_from_record({"props": dict(props)})
                card = card_for_row(row)
                if not card:
                    failed += 1
                    continue
                embedding = [float(x) for x in self.embedder.embed(card)]
                updated = self._set_claim_embedding(
                    graph,
                    pot_id=pot_id,
                    props=props,
                    embedding=embedding,
                    embedder_name=embedder_name,
                    embedder_dim=embedder_dim,
                )
                if updated < 1:
                    raise RuntimeError("re-embed matched no edge")
                repaired += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "semantic re-embed failed for %s:%s->%s: %s",
                    props.get("name"),
                    props.get("subject_key"),
                    props.get("object_key"),
                    exc,
                )
        # Recreate (or ensure) the vector index after values are rewritten.
        # The index covers every :RELATES_TO edge in the (shared) graph, so a
        # dimension migration re-shapes it for all pots in this store.
        assert self.writer is not None
        embedding_dim = int(getattr(self.embedder, "dimensions", 1536))
        if not FalkorDBGraphWriter._ensure_vector_index_sync(graph, embedding_dim):
            index_errors.append("index recreate failed — see the daemon/CLI log")
        index_recreated = index_recreated and not index_errors
        detail = (
            f"re-embedded {repaired} of {len(needs)} stale claim(s) "
            f"(scanned {len(props_list)})"
        )
        if failed:
            detail += f"; {failed} FAILED — see warnings in the daemon/CLI log"
        if index_recreated:
            detail += "; vector index rebuilt for new embedding dimensions"
        if index_errors:
            detail += "; " + "; ".join(index_errors)
        return {
            "scanned": len(props_list),
            "repaired": repaired,
            "failed": failed + (1 if index_errors else 0),
            "index_recreated": index_recreated,
            "detail": detail,
        }

    @staticmethod
    def _set_claim_embedding(
        graph: Any,
        *,
        pot_id: str,
        props: Mapping[str, Any],
        embedding: list[float],
        embedder_name: str,
        embedder_dim: int,
    ) -> int:
        claim_key = props.get("claim_key")
        base_params = {
            "gid": pot_id,
            "embedding": embedding,
            "embedding_model": embedder_name,
            "embedding_dim": embedder_dim,
        }
        if isinstance(claim_key, str) and claim_key:
            result = graph.query(
                _REEMBED_BY_CLAIM_KEY_CYPHER,
                params={**base_params, "claim_key": claim_key},
            )
        else:
            result = graph.query(
                _REEMBED_BY_TUPLE_CYPHER,
                params={
                    **base_params,
                    "predicate": props.get("name"),
                    "subject_key": props.get("subject_key"),
                    "object_key": props.get("object_key"),
                    "source_ref": props.get("source_ref"),
                },
            )
        records = _records_from_result(result)
        return int(records[0].get("updated") or 0) if records else 0

    def _repair_entity_summaries(self, pot_id: str) -> int:
        graph = self.graph_provider()
        rows = _records_from_result(
            graph.query(
                ENTITY_SUMMARY_SCAN_CYPHER,
                params={"gid": pot_id, "limit": ENTITY_SUMMARY_REPAIR_LIMIT},
            )
        )
        repaired = 0
        for row in rows:
            key = str(row.get("key") or "").strip()
            if not key:
                continue
            raw_props = row.get("props")
            fixed = repaired_entity_properties(
                key, raw_props if isinstance(raw_props, Mapping) else {}
            )
            if fixed is None:
                continue
            result = graph.query(
                ENTITY_SUMMARY_UPDATE_CYPHER,
                params={"gid": pot_id, "key": key, "props": fixed},
            )
            records = _records_from_result(result)
            if not records:
                repaired += 1
                continue
            repaired += int(records[0].get("cnt") or 0)
        return repaired


class FalkorDBLiteGraphBackend(FalkorDBGraphBackend):
    """FalkorDBLite-backed profile using the same Falkor adapter bundle."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("profile_name", _LITE_PROFILE)
        kwargs.setdefault("force_mode", "lite")
        super().__init__(*args, **kwargs)


__all__ = ["FalkorDBGraphBackend", "FalkorDBLiteGraphBackend"]
