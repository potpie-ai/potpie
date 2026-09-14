"""LadybugDB writer + connection provider (Claim-as-node physical model).

Ladybug is schema-first and only indexes vectors on **node** tables, so claims
live as ``:Claim`` nodes (with ``CLAIM_SUBJECT`` / ``CLAIM_OBJECT`` edges to
``:Entity``) rather than Falkor's ``:RELATES_TO`` relationships. Application
ports still speak ``GraphWriterPort`` / ``ClaimRow``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from potpie_context_engine.adapters.outbound.graph.cypher import (
    _render_fact,
    _require_valid_pot_id,
    _stable_source_ref,
)
from potpie_context_engine.adapters.outbound.graph.writer_port import GraphWriterPort
from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort
from potpie_context_engine.adapters.outbound.graph.ladybug_vector import (
    LadybugVectorConfig,
    VECTOR_INDEX_NAME,
    create_vector_index,
    pad_embedding,
    prepare_embedding,
)
from potpie_context_engine.domain.retrieval_card import build_retrieval_card

logger = logging.getLogger(__name__)

_VECTOR_INDEX = VECTOR_INDEX_NAME
_DEFAULT_DIM = 256


def _require_ladybug() -> Any:
    from potpie_context_engine.adapters.outbound.graph.ladybug_windows_bootstrap import (
        ensure_ladybug_openssl,
    )

    openssl = ensure_ladybug_openssl()
    if not openssl.get("ok"):
        raise ImportError(
            "Ladybug OpenSSL bootstrap failed on Windows: "
            f"{openssl.get('error', 'unknown')}. "
            "Use python.org CPython 3.12/3.13 x64 and reinstall "
            "'potpie-context-engine[local]'."
        )
    try:
        import ladybug as lb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Ladybug backend requires the 'ladybug' package. "
            "Install with: pip install 'potpie-context-engine[local]'"
        ) from exc
    return lb


def _now_ts() -> datetime:
    """UTC timestamp for Ladybug ``TIMESTAMP`` columns (no string cast)."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _as_ts(value: Any) -> datetime | None:
    """Coerce write params to ``datetime`` for Ladybug TIMESTAMP binders."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return None


def _claim_key(
    *,
    pot_id: str,
    predicate: str,
    from_key: str,
    to_key: str,
    source_ref: str,
) -> str:
    raw = f"{pot_id}|{predicate}|{from_key}|{to_key}|{source_ref}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _records_from_result(result: Any) -> list[dict[str, Any]]:
    """Normalize Ladybug query results to list[dict]."""
    if result is None:
        return []
    try:
        rows_as_dict = getattr(result, "rows_as_dict", None)
        if callable(rows_as_dict):
            try:
                wrapped = rows_as_dict()
                out = [dict(row) for row in wrapped]
                if out:
                    return out
            except Exception:
                pass
        get_names = getattr(result, "get_column_names", None)
        names = list(get_names()) if callable(get_names) else []
        rows = list(result)
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
            elif isinstance(row, (list, tuple)) and names and len(names) == len(row):
                out.append({str(n): v for n, v in zip(names, row)})
            elif isinstance(row, (list, tuple)):
                out.append({str(i): v for i, v in enumerate(row)})
            else:
                out.append({"value": row})
        return out
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _execute(conn: Any, cypher: str, params: Mapping[str, Any] | None = None) -> Any:
    if params:
        return conn.execute(cypher, dict(params))
    return conn.execute(cypher)


def build_ladybug_db(settings: ContextEngineSettingsPort) -> Any:
    """Open (or create) the on-disk Ladybug database and return a Connection."""
    lb = _require_ladybug()
    path = settings.ladybug_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    db = lb.Database(path)
    conn = lb.Connection(db)
    conn._potpie_db = db  # type: ignore[attr-defined]
    return conn


def close_ladybug_conn(conn: Any | None) -> None:
    """Close Connection then Database (required for Windows reopen)."""
    if conn is None:
        return
    db = getattr(conn, "_potpie_db", None)
    try:
        conn.close()
    except Exception:
        pass
    if db is not None:
        try:
            db.close()
        except Exception:
            pass


class LadybugGraphProvider:
    """Lazily build and memoize one shared Ladybug connection + database."""

    def __init__(self, settings: ContextEngineSettingsPort) -> None:
        self._settings = settings
        self._conn: Any | None = None
        self._db: Any | None = None

    def __call__(self) -> Any:
        if self._conn is None:
            lb = _require_ladybug()
            path = self._settings.ladybug_path()
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._db = lb.Database(path)
            self._conn = lb.Connection(self._db)
            self._conn._potpie_db = self._db  # type: ignore[attr-defined]
        return self._conn

    def close(self) -> None:
        close_ladybug_conn(self._conn)
        self._conn = None
        self._db = None


def ensure_schema(
    conn: Any,
    embedding_dim: int,
    *,
    vector_config: LadybugVectorConfig | None = None,
) -> bool:
    """Idempotent DDL + vector extension/index. Returns True on success."""
    dim = max(1, int(embedding_dim))
    # Detect existing schema via a cheap match; create only when missing.
    try:
        _execute(conn, "MATCH (e:Entity) RETURN count(e) AS c LIMIT 1")
        schema_exists = True
    except Exception:
        schema_exists = False

    if not schema_exists:
        _execute(
            conn,
            """
            CREATE NODE TABLE Entity(
              entity_key STRING PRIMARY KEY,
              group_id STRING,
              name STRING,
              summary STRING,
              extra_labels STRING[]
            );
            """,
        )
        _execute(
            conn,
            f"""
            CREATE NODE TABLE Claim(
              claim_key STRING PRIMARY KEY,
              group_id STRING,
              name STRING,
              subject_key STRING,
              object_key STRING,
              fact STRING,
              fact_embedding FLOAT[{dim}],
              embedding_model STRING,
              embedding_dim INT64,
              truth DOUBLE,
              confidence DOUBLE,
              source_ref STRING,
              source_system STRING,
              subgraph STRING,
              mutation_id STRING,
              valid_at TIMESTAMP,
              invalid_at TIMESTAMP,
              created_at TIMESTAMP,
              expired_at TIMESTAMP
            );
            """,
        )
        _execute(conn, "CREATE REL TABLE CLAIM_SUBJECT(FROM Claim TO Entity);")
        _execute(conn, "CREATE REL TABLE CLAIM_OBJECT(FROM Claim TO Entity);")

    cfg = vector_config or LadybugVectorConfig()
    try:
        create_vector_index(conn, cfg)
    except Exception as exc:
        logger.debug("CREATE_VECTOR_INDEX: %s", exc)
    return True


def _pad_embedding(values: list[float], dim: int) -> list[float]:
    return pad_embedding(values, dim)


class LadybugGraphWriter(GraphWriterPort):
    """GraphWriterPort over Ladybug Claim-as-node storage."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        conn: Any | None = None,
        conn_provider: Callable[[], Any] | None = None,
        embedder: Any | None = None,
        definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
    ) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._conn = conn
        self._conn_provider = conn_provider
        self._embedder = embedder
        self._definition = definition
        self._schema_ready = False
        self._vector_config = LadybugVectorConfig.from_settings(settings)

    def bind_definition(self, definition: GraphDefinition) -> LadybugGraphWriter:
        return LadybugGraphWriter(
            self._settings,
            conn=self._conn,
            conn_provider=self._conn_provider,
            embedder=self._embedder,
            definition=definition,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def _get_conn(self) -> Any:
        if self._conn_provider is not None:
            return self._conn_provider()
        if self._conn is None:
            self._conn = build_ladybug_db(self._settings)
        return self._conn

    def _embedding_dim(self) -> int:
        return int(getattr(self._embedder, "dimensions", _DEFAULT_DIM) or _DEFAULT_DIM)

    async def ensure_indexes(self) -> bool:
        if not self.enabled:
            return False
        conn = self._get_conn()
        dim = self._embedding_dim()
        await asyncio.to_thread(
            ensure_schema, conn, dim, vector_config=self._vector_config
        )
        self._schema_ready = True
        return True

    def _ensure_schema_sync(self) -> None:
        if self._schema_ready:
            return
        ensure_schema(
            self._get_conn(),
            self._embedding_dim(),
            vector_config=self._vector_config,
        )
        self._schema_ready = True

    async def upsert_entities(
        self, pot_id: str, items: list[EntityUpsert], provenance: ProvenanceRef
    ) -> int:
        _require_valid_pot_id(pot_id)
        if not items:
            return 0
        await self.ensure_indexes()
        conn = self._get_conn()
        n = 0
        for item in items:
            labels = [str(x) for x in (item.labels or ()) if str(x) and str(x) != "Entity"]
            name = str(item.properties.get("name") or item.entity_key)
            summary = str(item.properties.get("summary") or "")
            await asyncio.to_thread(
                _execute,
                conn,
                """
                MERGE (e:Entity {entity_key: $key})
                ON CREATE SET e.group_id = $gid, e.name = $name,
                              e.summary = $summary, e.extra_labels = $labs
                ON MATCH SET e.group_id = $gid, e.name = $name,
                             e.summary = $summary, e.extra_labels = $labs
                """,
                {
                    "key": item.entity_key,
                    "gid": pot_id,
                    "name": name,
                    "summary": summary,
                    "labs": labels,
                },
            )
            n += 1
        return n

    async def upsert_edges(
        self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
    ) -> int:
        _require_valid_pot_id(pot_id)
        if not items:
            return 0
        await self.ensure_indexes()
        conn = self._get_conn()
        dim = self._embedding_dim()
        now = _now_ts()
        n = 0
        prepared: list[tuple[Any, dict[str, Any], str, str, str, str]] = []
        cards: list[str] = []
        for item in items:
            predicate = item.edge_type or "RELATED_TO"
            source_ref = _stable_source_ref(
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                provenance=provenance,
            )
            props = dict(item.properties or {})
            if isinstance(props.get("source_ref"), str) and props["source_ref"]:
                source_ref = props["source_ref"]
            fact = props.get("fact") or _render_fact(
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                extra=props,
            )
            ck = props.get("claim_key") or _claim_key(
                pot_id=pot_id,
                predicate=predicate,
                from_key=item.from_entity_key,
                to_key=item.to_entity_key,
                source_ref=source_ref,
            )
            card = ""
            if self._embedder is not None:
                card = build_retrieval_card(
                    description=str(props.get("description") or ""),
                    fact=str(fact),
                    subject_key=item.from_entity_key,
                    object_key=item.to_entity_key,
                    predicate=predicate,
                )
            cards.append(card)
            prepared.append((item, props, predicate, source_ref, fact, ck))

        embeddings: list[list[float] | None] = [None] * len(prepared)
        model = None
        if self._embedder is not None and cards:
            try:
                embed_many = getattr(self._embedder, "embed_many", None)
                if callable(embed_many) and len(cards) > 1:
                    raw_vecs = embed_many(cards)
                else:
                    raw_vecs = [self._embedder.embed(c) for c in cards]
                model = getattr(self._embedder, "model_name", None) or "local"
                for i, raw in enumerate(raw_vecs):
                    embeddings[i] = prepare_embedding(
                        [float(x) for x in raw],
                        dim=dim,
                        config=self._vector_config,
                    )
            except Exception:
                logger.debug("batch embed failed; per-claim fallback", exc_info=True)
                for i, card in enumerate(cards):
                    try:
                        embeddings[i] = prepare_embedding(
                            [float(x) for x in self._embedder.embed(card)],
                            dim=dim,
                            config=self._vector_config,
                        )
                    except Exception:
                        logger.debug(
                            "embed failed for claim %s",
                            prepared[i][5],
                            exc_info=True,
                        )

        for (item, props, predicate, source_ref, fact, ck), emb in zip(
            prepared, embeddings
        ):
            if emb is None:
                emb = [0.0] * dim

            conf = props.get("confidence")
            confidence = float(conf) if conf is not None else None
            valid_at = _as_ts(props.get("valid_at")) or now
            await asyncio.to_thread(
                self._upsert_claim_sync,
                conn,
                {
                    "ck": ck,
                    "gid": pot_id,
                    "pred": predicate,
                    "sk": item.from_entity_key,
                    "ok": item.to_entity_key,
                    "fact": str(fact),
                    "emb": emb,
                    "model": model,
                    "dim": dim,
                    "truth": float(props["truth"])
                    if isinstance(props.get("truth"), (int, float))
                    else None,
                    "confidence": confidence,
                    "src": source_ref,
                    "sys": props.get("source_system")
                    or provenance.source_system
                    or "agent",
                    "subgraph": props.get("subgraph"),
                    "mutation_id": provenance.mutation_id,
                    "valid_at": valid_at,
                    "now": now,
                },
            )
            if predicate in self._definition.singleton_predicates:
                await asyncio.to_thread(
                    self._supersede_sync,
                    conn,
                    pot_id,
                    predicate,
                    item.from_entity_key,
                    item.to_entity_key,
                    ck,
                    now,
                )
            n += 1
        return n

    def _upsert_claim_sync(self, conn: Any, p: dict[str, Any]) -> None:
        # Ensure endpoints exist (writer may upsert edges after entities).
        for key in (p["sk"], p["ok"]):
            _execute(
                conn,
                """
                MERGE (e:Entity {entity_key: $key})
                ON CREATE SET e.group_id = $gid, e.name = $key,
                              e.summary = '', e.extra_labels = []
                """,
                {"key": key, "gid": p["gid"]},
            )
        _execute(
            conn,
            """
            MERGE (c:Claim {claim_key: $ck})
            ON CREATE SET
              c.group_id = $gid, c.name = $pred,
              c.subject_key = $sk, c.object_key = $ok,
              c.fact = $fact, c.fact_embedding = $emb,
              c.embedding_model = $model, c.embedding_dim = $dim,
              c.truth = $truth, c.confidence = $confidence,
              c.source_ref = $src, c.source_system = $sys,
              c.subgraph = $subgraph, c.mutation_id = $mutation_id,
              c.valid_at = $valid_at, c.invalid_at = null,
              c.created_at = $now, c.expired_at = null
            ON MATCH SET
              c.group_id = $gid, c.name = $pred,
              c.subject_key = $sk, c.object_key = $ok,
              c.fact = $fact, c.fact_embedding = $emb,
              c.embedding_model = $model, c.embedding_dim = $dim,
              c.truth = $truth, c.confidence = $confidence,
              c.source_ref = $src, c.source_system = $sys,
              c.subgraph = $subgraph, c.mutation_id = $mutation_id,
              c.valid_at = $valid_at, c.invalid_at = null
            """,
            p,
        )
        # Refresh structural edges (delete+create is simplest / idempotent enough).
        _execute(
            conn,
            "MATCH (c:Claim {claim_key: $ck})-[r:CLAIM_SUBJECT]->() DELETE r",
            {"ck": p["ck"]},
        )
        _execute(
            conn,
            "MATCH (c:Claim {claim_key: $ck})-[r:CLAIM_OBJECT]->() DELETE r",
            {"ck": p["ck"]},
        )
        _execute(
            conn,
            "MATCH (c:Claim {claim_key: $ck}), (e:Entity {entity_key: $ek}) "
            "CREATE (c)-[:CLAIM_SUBJECT]->(e)",
            {"ck": p["ck"], "ek": p["sk"]},
        )
        _execute(
            conn,
            "MATCH (c:Claim {claim_key: $ck}), (e:Entity {entity_key: $ek}) "
            "CREATE (c)-[:CLAIM_OBJECT]->(e)",
            {"ck": p["ck"], "ek": p["ok"]},
        )

    def _supersede_sync(
        self,
        conn: Any,
        pot_id: str,
        predicate: str,
        subject: str,
        object_key: str,
        keep_ck: str,
        now: str,
    ) -> None:
        _execute(
            conn,
            """
            MATCH (c:Claim)
            WHERE c.group_id = $gid AND c.name = $pred
              AND c.subject_key = $sk AND c.object_key <> $ok
              AND c.invalid_at IS NULL AND c.claim_key <> $keep
            SET c.invalid_at = $now, c.expired_at = $now
            """,
            {
                "gid": pot_id,
                "pred": predicate,
                "sk": subject,
                "ok": object_key,
                "keep": keep_ck,
                "now": now,
            },
        )

    async def delete_edges(
        self, pot_id: str, items: list[EdgeDelete], provenance: ProvenanceRef
    ) -> int:
        _require_valid_pot_id(pot_id)
        if not items:
            return 0
        await self.ensure_indexes()
        conn = self._get_conn()
        n = 0
        for item in items:
            result = await asyncio.to_thread(
                _execute,
                conn,
                """
                MATCH (c:Claim)
                WHERE c.group_id = $gid AND c.name = $pred
                  AND c.subject_key = $sk AND c.object_key = $ok
                DETACH DELETE c
                RETURN count(*) AS cnt
                """,
                {
                    "gid": pot_id,
                    "pred": item.edge_type,
                    "sk": item.from_entity_key,
                    "ok": item.to_entity_key,
                },
            )
            rows = _records_from_result(result)
            if rows:
                row = rows[0]
                n += int(row.get("cnt") or row.get("0") or 0)
            else:
                n += 1
        return n

    async def invalidate(
        self, pot_id: str, items: list[InvalidationOp], provenance: ProvenanceRef
    ) -> int:
        _require_valid_pot_id(pot_id)
        if not items:
            return 0
        await self.ensure_indexes()
        conn = self._get_conn()
        now = _now_ts()
        n = 0
        for item in items:
            if item.target_edge:
                etype, frm, to = item.target_edge
                await asyncio.to_thread(
                    _execute,
                    conn,
                    """
                    MATCH (c:Claim)
                    WHERE c.group_id = $gid AND c.name = $pred
                      AND c.subject_key = $sk AND c.object_key = $ok
                      AND c.invalid_at IS NULL
                    SET c.invalid_at = $now, c.expired_at = $now
                    """,
                    {
                        "gid": pot_id,
                        "pred": etype,
                        "sk": frm,
                        "ok": to,
                        "now": now,
                    },
                )
                n += 1
            elif item.target_entity_key:
                await asyncio.to_thread(
                    _execute,
                    conn,
                    """
                    MATCH (c:Claim)
                    WHERE c.group_id = $gid
                      AND (c.subject_key = $key OR c.object_key = $key)
                      AND c.invalid_at IS NULL
                    SET c.invalid_at = $now, c.expired_at = $now
                    """,
                    {"gid": pot_id, "key": item.target_entity_key, "now": now},
                )
                n += 1
        return n

    async def invalidate_claim_keys(
        self,
        pot_id: str,
        claim_keys: list[str] | tuple[str, ...],
        *,
        reason: str | None = None,
    ) -> int:
        """Soft-invalidate live claims by their exact ``claim_key`` values."""
        del reason  # stamped via invalid_at only; reason is API parity with the port
        _require_valid_pot_id(pot_id)
        keys = [k for k in claim_keys if isinstance(k, str) and k.strip()]
        if not keys:
            return 0
        await self.ensure_indexes()
        conn = self._get_conn()
        now = _now_ts()

        def _run() -> int:
            rows = _records_from_result(
                _execute(
                    conn,
                    """
                    MATCH (c:Claim)
                    WHERE c.group_id = $gid
                      AND c.invalid_at IS NULL
                      AND c.claim_key IN $keys
                    SET c.invalid_at = $now, c.expired_at = $now
                    RETURN count(c) AS cnt
                    """,
                    {"gid": pot_id, "keys": keys, "now": now},
                )
            )
            if not rows:
                return 0
            row = rows[0]
            return int(row.get("cnt") or row.get("0") or 0)

        return await asyncio.to_thread(_run)

    async def reset_pot(self, pot_id: str) -> dict[str, Any]:
        _require_valid_pot_id(pot_id)
        await self.ensure_indexes()
        conn = self._get_conn()

        def _count() -> int:
            rows = _records_from_result(
                _execute(
                    conn,
                    """
                    MATCH (n)
                    WHERE n.group_id = $gid
                    RETURN count(n) AS cnt
                    """,
                    {"gid": pot_id},
                )
            )
            if not rows:
                return 0
            row = rows[0]
            return int(row.get("cnt") or row.get("0") or 0)

        def _reset() -> dict[str, Any]:
            before = _count()
            _execute(
                conn,
                "MATCH (c:Claim) WHERE c.group_id = $gid DETACH DELETE c",
                {"gid": pot_id},
            )
            _execute(
                conn,
                "MATCH (e:Entity) WHERE e.group_id = $gid DETACH DELETE e",
                {"gid": pot_id},
            )
            remaining = _count()
            return {
                "ok": remaining == 0,
                "pot_id": pot_id,
                "group_id_nodes_before": before,
                "group_id_nodes_remaining": remaining,
                **(
                    {}
                    if remaining == 0
                    else {"error": "group_id_reset_incomplete"}
                ),
            }

        return await asyncio.to_thread(_reset)


__all__ = [
    "LadybugGraphProvider",
    "LadybugGraphWriter",
    "build_ladybug_db",
    "close_ladybug_conn",
    "ensure_schema",
    "_pad_embedding",
    "_records_from_result",
    "_require_ladybug",
    "_VECTOR_INDEX",
]
