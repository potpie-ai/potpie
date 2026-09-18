"""Ladybug native HNSW configuration and embedding helpers (OSS defaults).

Tuning knobs map to Ladybug ``CREATE_VECTOR_INDEX`` / ``QUERY_VECTOR_INDEX``:
https://docs.ladybugdb.com/extensions/vector/

Defaults target high recall on normalized embeddings (cosine metric) while
keeping query latency in the tens-of-ms range on desktop CLI workloads.
Override via ``CONTEXT_ENGINE_LADYBUG_VECTOR_*`` env vars.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

VECTOR_INDEX_NAME = "claim_fact_embedding"

# Oversample ANN candidates when post-filtering in Python.
VECTOR_OVERSAMPLE_K = 50
VECTOR_OVERSAMPLE_FACTOR = 5


def _env_int(name: str, default: int, *, lo: int = 1, hi: int = 10_000) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, v))


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("0", "false", "no", "off"):
        return False
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class LadybugVectorConfig:
    """HNSW index build + query parameters."""

    metric: str = "cosine"
    mu: int = 30
    ml: int = 60
    pu: float = 0.05
    efc: int = 200
    efs: int = 200
    cache_embeddings: bool = True
    normalize_embeddings: bool = True

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> LadybugVectorConfig:
        if settings is not None:
            return cls(
                metric=getattr(settings, "ladybug_vector_metric", lambda: "cosine")(),
                mu=int(getattr(settings, "ladybug_vector_mu", lambda: 30)()),
                ml=int(getattr(settings, "ladybug_vector_ml", lambda: 60)()),
                pu=float(getattr(settings, "ladybug_vector_pu", lambda: 0.05)()),
                efc=int(getattr(settings, "ladybug_vector_efc", lambda: 200)()),
                efs=int(getattr(settings, "ladybug_vector_efs", lambda: 200)()),
                cache_embeddings=bool(
                    getattr(settings, "ladybug_vector_cache_embeddings", lambda: True)()
                ),
                normalize_embeddings=bool(
                    getattr(
                        settings, "ladybug_vector_normalize_embeddings", lambda: True
                    )()
                ),
            )
        return cls(
            mu=_env_int("CONTEXT_ENGINE_LADYBUG_VECTOR_MU", 30, lo=4, hi=256),
            ml=_env_int("CONTEXT_ENGINE_LADYBUG_VECTOR_ML", 60, lo=8, hi=512),
            efc=_env_int("CONTEXT_ENGINE_LADYBUG_VECTOR_EFC", 200, lo=32, hi=2000),
            efs=_env_int("CONTEXT_ENGINE_LADYBUG_VECTOR_EFS", 200, lo=32, hi=2000),
            cache_embeddings=_env_bool(
                "CONTEXT_ENGINE_LADYBUG_VECTOR_CACHE_EMBEDDINGS", True
            ),
            normalize_embeddings=_env_bool(
                "CONTEXT_ENGINE_LADYBUG_VECTOR_NORMALIZE", True
            ),
        )

    def create_index_cypher(self, *, index_name: str = VECTOR_INDEX_NAME) -> str:
        cache = "true" if self.cache_embeddings else "false"
        return (
            "CALL CREATE_VECTOR_INDEX("
            f"'Claim', '{index_name}', 'fact_embedding', "
            f"mu := {self.mu}, ml := {self.ml}, pu := {self.pu}, "
            f"metric := '{self.metric}', efc := {self.efc}, "
            f"cache_embeddings := {cache})"
        )

    def ann_k(self, limit: int) -> int:
        """Candidate pool size for ANN before structural post-filters."""
        lim = max(1, int(limit))
        return max(lim * VECTOR_OVERSAMPLE_FACTOR, VECTOR_OVERSAMPLE_K)


def normalize_embedding(values: list[float]) -> list[float]:
    """L2-normalize for cosine metric (no-op on zero vectors)."""
    if not values:
        return values
    norm = math.sqrt(sum(x * x for x in values))
    if norm <= 1e-12:
        return values
    inv = 1.0 / norm
    return [float(x) * inv for x in values]


def prepare_embedding(
    values: list[float],
    *,
    dim: int,
    config: LadybugVectorConfig,
) -> list[float]:
    """Pad/truncate then optionally normalize for index + query."""
    vec = pad_embedding(values, dim)
    if config.normalize_embeddings:
        vec = normalize_embedding(vec)
    return vec


def pad_embedding(values: list[float], dim: int) -> list[float]:
    if len(values) == dim:
        return values
    if len(values) > dim:
        return values[:dim]
    return values + [0.0] * (dim - len(values))


def load_vector_extension(conn: Any) -> None:
    for stmt in ("INSTALL VECTOR;", "LOAD VECTOR;"):
        try:
            conn.execute(stmt)
        except Exception:
            pass


def create_vector_index(conn: Any, config: LadybugVectorConfig) -> None:
    load_vector_extension(conn)
    try:
        conn.execute(config.create_index_cypher())
    except Exception:
        # Index may already exist on reprovision.
        pass


def query_vector_index(
    conn: Any,
    *,
    embedding: list[float],
    k: int,
    config: LadybugVectorConfig,
    index_name: str = VECTOR_INDEX_NAME,
) -> Any:
    load_vector_extension(conn)
    return conn.execute(
        f"""
        CALL QUERY_VECTOR_INDEX(
            'Claim', '{index_name}', $embedding, $k, efs := $efs
        )
        RETURN node.claim_key AS claim_key, node.group_id AS group_id,
               node.name AS name, node.subject_key AS subject_key,
               node.object_key AS object_key, node.fact AS fact,
               node.truth AS truth, node.confidence AS confidence,
               node.source_ref AS source_ref, node.source_system AS source_system,
               node.subgraph AS subgraph, node.mutation_id AS mutation_id,
               node.valid_at AS valid_at, node.invalid_at AS invalid_at,
               node.created_at AS created_at, node.expired_at AS expired_at,
               node.embedding_model AS embedding_model,
               node.embedding_dim AS embedding_dim,
               node.fact_embedding AS fact_embedding,
               distance
        ORDER BY distance
        """,
        {"embedding": embedding, "k": int(k), "efs": int(config.efs)},
    )


def has_graph_prune_filters(params: Mapping[str, object]) -> bool:
    """True when Cypher filters shrink the search space (hybrid graph-first path)."""
    return bool(
        params.get("has_preds")
        or params.get("has_subjects")
        or params.get("has_objects")
        or params.get("has_claim_keys")
        or params.get("has_subgraphs")
        or params.get("has_mutation_ids")
        or params.get("has_source_refs")
        or params.get("has_sources")
    )


__all__ = [
    "LadybugVectorConfig",
    "VECTOR_INDEX_NAME",
    "create_vector_index",
    "has_graph_prune_filters",
    "load_vector_extension",
    "normalize_embedding",
    "pad_embedding",
    "prepare_embedding",
    "query_vector_index",
]
