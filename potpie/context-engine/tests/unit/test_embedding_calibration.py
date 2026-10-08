"""Absolute similarity thresholds apply only to the model they were measured on.

``SIMILARITY_BLEND`` and the relevance confidence bands were measured with
``all-MiniLM-L6-v2``. An embedder may call itself ``calibrated`` only for that
model; every other embedder takes the rank-and-coverage fallback the resources
reader already implements for ``similarity_calibrated=False``.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from potpie_context_engine.adapters.outbound.resources.index import (
    SqliteFtsResourceIndex,
    SqliteHybridResourceIndex,
)
from potpie_context_engine.core.ports.resource_index import (
    CALIBRATED_EMBEDDING_MODELS,
    embedding_model_is_calibrated,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("model", "calibrated"),
    [
        ("all-MiniLM-L6-v2", True),
        ("sentence-transformers/all-MiniLM-L6-v2", True),
        ("  ALL-MINILM-L6-V2 ", True),
        ("all-MiniLM-L12-v2", False),
        ("all-mpnet-base-v2", False),
        ("BAAI/bge-small-en-v1.5", False),
        ("", False),
        (None, False),
    ],
)
def test_only_the_measured_model_is_calibrated(model, calibrated) -> None:
    assert embedding_model_is_calibrated(model) is calibrated


@pytest.mark.parametrize(
    ("model", "calibrated"),
    [("all-MiniLM-L6-v2", True), ("all-mpnet-base-v2", False)],
)
def test_sentence_transformer_calibration_follows_its_model(model, calibrated) -> None:
    embedder = SentenceTransformerEmbedder(model_name=model)

    assert embedder.calibrated is calibrated
    # Declaring calibration must not load the model.
    assert embedder._model is None


def test_the_default_model_is_the_measured_one() -> None:
    """Changing the default model without re-measuring would silently drop
    every deployment onto the uncalibrated fallback."""
    assert embedding_model_is_calibrated(DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    assert CALIBRATED_EMBEDDING_MODELS == frozenset({"all-minilm-l6-v2"})


def test_the_hashing_embedder_is_never_calibrated() -> None:
    assert HashingEmbedder().calibrated is False


@pytest.mark.parametrize(
    ("model", "calibrated"),
    [("all-MiniLM-L6-v2", True), ("all-mpnet-base-v2", False)],
)
def test_the_index_reports_its_embedders_calibration(
    tmp_path, model, calibrated
) -> None:
    index = SqliteHybridResourceIndex(
        embedder=SentenceTransformerEmbedder(model_name=model), home=tmp_path
    )

    assert index.similarity_calibrated() is calibrated


def test_a_lexical_index_is_never_calibrated(tmp_path) -> None:
    assert SqliteFtsResourceIndex(home=tmp_path).similarity_calibrated() is False
