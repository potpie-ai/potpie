"""Setup-time embedding model selection and fallback behavior."""

from __future__ import annotations

import json
import logging
import os
import sys
import types
import warnings
from unittest.mock import MagicMock

import pytest

from potpie_context_engine.adapters.outbound.intelligence import local_embedder
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from potpie.services.config_service import LocalConfigService
from potpie.services.setup_orchestrator import DefaultSetupOrchestrator
from potpie_context_core.lifecycle import DONE, FAILED, SetupPlan

pytestmark = pytest.mark.unit


def _clear_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CONTEXT_ENGINE_EMBEDDER",
        "CONTEXT_ENGINE_EMBEDDING_MODEL",
        "CONTEXT_ENGINE_EMBEDDING_CACHE",
        "CONTEXT_ENGINE_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_no_config_defaults_to_bundled_hashing_embedder(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    _clear_embedding_env(monkeypatch)

    embedder = local_embedder.build_embedder()

    assert isinstance(embedder, HashingEmbedder)
    assert embedder.name == "local-hashing-v1"


def test_build_embedder_reads_setup_sentence_transformer_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models" / "sentence-transformers"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "embedder": "sentence-transformers",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_cache": str(cache),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    _clear_embedding_env(monkeypatch)
    monkeypatch.setattr(
        local_embedder, "_sentence_transformers_installed", lambda: True
    )

    embedder = local_embedder.build_embedder()

    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.cache_folder == str(cache)


@pytest.mark.parametrize(
    "choice",
    [
        "auto",
        "best",
        "semantic",
        "legacy",
        "sentence-transformers",
        "sentence_transformers",
        "sbert",
        "minilm",
        "all-minilm-l6-v2",
    ],
)
def test_semantic_embedder_aliases_fall_back_to_hashing_when_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", choice)
    monkeypatch.setattr(
        local_embedder, "_sentence_transformers_installed", lambda: False
    )

    embedder = local_embedder.build_embedder()

    assert isinstance(embedder, HashingEmbedder)


def test_known_sentence_transformer_dimension_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_self):
        raise AssertionError("dimensions should not load the model")

    monkeypatch.setattr(SentenceTransformerEmbedder, "_get_model", fail_load)

    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    assert embedder.dimensions == 384


def test_sentence_transformer_prepare_suppresses_nested_progress(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class _FakeEmbedding:
        def tolist(self) -> list[float]:
            return [0.1, 0.2, 0.3]

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            calls["init_progress_env"] = os.getenv("HF_HUB_DISABLE_PROGRESS_BARS")

        def encode(self, text: str, **kwargs: object) -> _FakeEmbedding:
            calls["encode_text"] = text
            calls["encode_kwargs"] = kwargs
            calls["encode_progress_env"] = os.getenv("HF_HUB_DISABLE_PROGRESS_BARS")
            return _FakeEmbedding()

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    embedder = SentenceTransformerEmbedder(
        model_name="test-model",
        cache_folder=str(tmp_path),
    )
    metadata = embedder.prepare()

    assert calls["model_name"] == "test-model"
    assert calls["init_progress_env"] == "1"
    assert calls["encode_progress_env"] == "1"
    assert calls["encode_kwargs"] == {"show_progress_bar": False}
    assert metadata["dimensions"] == 3
    assert os.getenv("HF_HUB_DISABLE_PROGRESS_BARS") is None


def test_quiet_transformer_progress_suppresses_loader_noise(
    recwarn: pytest.WarningsRecorder,
) -> None:
    records: list[logging.LogRecord] = []
    noisy_logger = logging.getLogger("sentence_transformers.base.model")

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler()
    old_level = noisy_logger.level
    noisy_logger.setLevel(logging.INFO)
    noisy_logger.addHandler(handler)
    try:
        with local_embedder._quiet_transformer_progress():
            noisy_logger.warning("Loading SentenceTransformer model")
            warnings.warn(
                "Warning: You are sending unauthenticated requests to the HF Hub.",
                UserWarning,
                stacklevel=1,
            )
    finally:
        noisy_logger.removeHandler(handler)
        noisy_logger.setLevel(old_level)

    assert records == []
    assert list(recwarn) == []


def test_setup_config_persists_embedding_defaults(tmp_path) -> None:
    service = LocalConfigService(home=tmp_path)

    service.write_defaults(
        SetupPlan(
            embeddings="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
        )
    )

    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert data["embedder"] == "sentence-transformers"
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["embedding_cache"] == str(tmp_path / "models" / "sentence-transformers")


def test_setup_reports_semantic_alias_hashing_fallback() -> None:
    orchestrator = _setup_orchestrator_with_embedder(HashingEmbedder())

    step = orchestrator._embedding_model(SetupPlan(embeddings="auto"))  # noqa: SLF001

    assert step.state == FAILED
    assert step.detail == "sentence-transformers is unavailable; using local-hashing-v1"
    assert step.metadata["fallback"] == "local-hashing-v1"


def test_setup_embedding_model_metadata_preserves_resolved_model() -> None:
    class _PreparedEmbedder:
        name = "prepared-embedder"

        def prepare(self) -> dict[str, object]:
            return {"model": "", "cache_folder": None}

    orchestrator = _setup_orchestrator_with_embedder(_PreparedEmbedder())

    step = orchestrator._embedding_model(  # noqa: SLF001
        SetupPlan(embeddings="sentence-transformers", embedding_model="fallback-model")
    )

    assert step.state == DONE
    assert step.detail == "fallback-model ready"
    assert step.metadata["model"] == "fallback-model"


def _setup_orchestrator_with_embedder(embedder: object) -> DefaultSetupOrchestrator:
    backend = MagicMock()
    backend.embedder = embedder
    return DefaultSetupOrchestrator(
        config=MagicMock(),
        installer=MagicMock(),
        backend=backend,
        pots=MagicMock(),
        state_store=MagicMock(),
        migrator=MagicMock(),
        daemon=MagicMock(),
        auth=MagicMock(),
        skills=MagicMock(),
    )
