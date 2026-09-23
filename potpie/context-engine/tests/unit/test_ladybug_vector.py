"""Unit tests for Ladybug vector config and embedding helpers."""

from __future__ import annotations

import math

import pytest

from potpie_context_engine.adapters.outbound.graph.ladybug_vector import (
    LadybugVectorConfig,
    has_graph_prune_filters,
    normalize_embedding,
    prepare_embedding,
)

pytestmark = pytest.mark.unit


def test_normalize_embedding_unit_length() -> None:
    vec = normalize_embedding([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)


def test_prepare_embedding_pads_and_normalizes() -> None:
    cfg = LadybugVectorConfig(normalize_embeddings=True)
    out = prepare_embedding([1.0, 0.0], dim=4, config=cfg)
    assert len(out) == 4
    assert math.isclose(out[0], 1.0, rel_tol=1e-6)


def test_create_index_cypher_includes_tuning_params() -> None:
    cfg = LadybugVectorConfig(mu=32, ml=64, efc=400, cache_embeddings=True)
    cypher = cfg.create_index_cypher()
    assert "mu := 32" in cypher
    assert "ml := 64" in cypher
    assert "efc := 400" in cypher
    assert "cache_embeddings := true" in cypher
    assert "metric := 'cosine'" in cypher


def test_has_graph_prune_filters() -> None:
    assert has_graph_prune_filters({"has_preds": True})
    assert not has_graph_prune_filters({"has_preds": False, "has_subjects": False})


class _Settings:
    def ladybug_vector_efs(self) -> int:
        return 128

    def ladybug_vector_efc(self) -> int:
        return 300

    def ladybug_vector_mu(self) -> int:
        return 40

    def ladybug_vector_ml(self) -> int:
        return 80

    def ladybug_vector_pu(self) -> float:
        return 0.1

    def ladybug_vector_metric(self) -> str:
        return "cosine"

    def ladybug_vector_cache_embeddings(self) -> bool:
        return False

    def ladybug_vector_normalize_embeddings(self) -> bool:
        return True


def test_config_from_settings() -> None:
    cfg = LadybugVectorConfig.from_settings(_Settings())
    assert cfg.efs == 128
    assert cfg.efc == 300
    assert cfg.mu == 40
    assert cfg.cache_embeddings is False
