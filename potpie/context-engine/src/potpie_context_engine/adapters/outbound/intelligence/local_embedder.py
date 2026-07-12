"""Context-graph embedders.

The Trigger Model non-negotiable: retrieval must work with **no API key and no
external embeddings service**, on the user's machine. This is the default OSS
embedder. It is a real local embedding model in the feature-hashing family: text
is tokenized into words *and* character n-grams, each feature is hashed (via a
stable digest, so embeddings are identical across processes — required for the
JSON-persisted embedded backend) into a fixed-dimension accumulator with a signed
contribution, then L2-normalized.

Why this beats the old Jaccard stub for V1.5 recall:

- It scores in continuous vector space, so near-misses rank above noise instead
  of collapsing to 0.
- Character n-grams catch morphological variants (``retry`` / ``retries`` /
  ``retrying`` share trigrams) and typos — the paraphrase failure mode Jaccard
  misses entirely.

It is intentionally swappable: ``build_embedder`` reads ``CONTEXT_ENGINE_EMBEDDER``
first, then Potpie's local setup config. Before setup, the dependency-free hashing
embedder remains the no-config fallback. ``potpie setup`` can persist
``sentence-transformers`` so normal ingestion/search uses the stronger local model.
``CONTEXT_ENGINE_EMBEDDER=none`` disables embeddings and the store falls back to a
*labeled* lexical match.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from potpie_context_engine.domain.embedding_modes import (
    AUTO_SENTENCE_TRANSFORMER_ALIASES,
    DISABLED_EMBEDDER_ALIASES,
    EXPLICIT_SENTENCE_TRANSFORMER_ALIASES,
    HASHING_EMBEDDER_ALIASES,
    SEMANTIC_EMBEDDER_ALIASES,
    normalize_embedding_mode,
)
from potpie_context_engine.domain.ports.embedder import EmbedderPort

_DEFAULT_DIMENSIONS = 256
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
_WORD_RE = re.compile(r"[a-z0-9]+")
_CHAR_NGRAM_MIN = 3
_CHAR_NGRAM_MAX = 4
_KNOWN_SENTENCE_TRANSFORMER_DIMS = {
    "all-minilm-l6-v2": 384,
    "sentence-transformers/all-minilm-l6-v2": 384,
}
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic feature-hashing embedder (the bundled local default)."""

    dimensions: int = _DEFAULT_DIMENSIONS
    name: str = "local-hashing-v1"

    def __post_init__(self) -> None:
        # A non-positive dimension would crash at runtime (modulo-by-zero in
        # ``_bucket`` for 0, negative indexing for <0); reject it up front.
        self.dimensions = int(self.dimensions)
        if self.dimensions <= 0:
            raise ValueError("HashingEmbedder.dimensions must be a positive integer")

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dimensions
        for token, weight in _features(text):
            idx, sign = _bucket(token, self.dimensions)
            vec[idx] += sign * weight
        return _l2_normalize(vec)

    def embed_many(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self.embed(t) for t in texts]


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Legacy-compatible local embedder backed by ``sentence-transformers``."""

    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
    device: str = "cpu"
    cache_folder: str | None = None
    _model: object | None = None

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self.model_name}"

    @property
    def dimensions(self) -> int:
        if self._model is None:
            configured = _configured_embedding_dimensions()
            if configured is not None:
                return configured
        known = _known_sentence_transformer_dimensions(self.model_name)
        if known is not None:
            return known
        model = self._get_model()
        get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", lambda: None
        )
        dim = get_dim()
        return int(dim or 384)

    def embed(self, text: str) -> tuple[float, ...]:
        embedding = self._encode(text or "")
        return tuple(float(x) for x in embedding.tolist())

    def embed_many(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        embeddings = self._encode(list(texts))
        return [tuple(float(x) for x in emb.tolist()) for emb in embeddings]

    def _get_model(self):
        if self._model is None:
            with _quiet_transformer_progress():
                try:
                    from sentence_transformers import SentenceTransformer
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "CONTEXT_ENGINE_EMBEDDER=legacy requires sentence-transformers. "
                        "Install the legacy Potpie dependencies or add an embeddings extra."
                    ) from exc
                kwargs = {"device": self.device}
                if self.cache_folder:
                    kwargs["cache_folder"] = str(Path(self.cache_folder).expanduser())
                self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _encode(self, inputs: str | list[str]) -> Any:
        with _quiet_transformer_progress():
            model = self._get_model()
            try:
                return model.encode(inputs, show_progress_bar=False)
            except TypeError as exc:
                if "show_progress_bar" not in str(exc):
                    raise
                return model.encode(inputs)

    def prepare(self) -> dict[str, object]:
        """Download/load the model and verify a small embedding end to end."""
        probe = self.embed("potpie context graph semantic search")
        return {
            "provider": "sentence-transformers",
            "model": self.model_name,
            "dimensions": len(probe),
            "cache_folder": str(Path(self.cache_folder).expanduser())
            if self.cache_folder
            else None,
        }


def _features(text: str):
    """Yield (token, weight) features: whole words plus char n-grams.

    Words carry full weight; subword n-grams carry a smaller weight so exact
    word matches dominate but morphological variants still contribute.
    """
    if not text:
        return
    words = _WORD_RE.findall(text.lower())
    for word in words:
        yield (f"w:{word}", 1.0)
        # Bound the n-gram cost on very long tokens.
        padded = f"^{word}$"
        for n in range(_CHAR_NGRAM_MIN, _CHAR_NGRAM_MAX + 1):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                yield (f"c{n}:{padded[i : i + n]}", 0.45)


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Stable bucket index + sign for a feature (process-independent hash)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    idx = value % dimensions
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return idx, sign


def _l2_normalize(vec: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return tuple(vec)
    return tuple(x / norm for x in vec)


@contextmanager
def _quiet_transformer_progress() -> Iterator[None]:
    """Let Potpie's setup UI own progress instead of nested third-party bars."""
    previous_hf_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    previous_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    previous_logging_disable = logging.root.manager.disable
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.disable(max(previous_logging_disable, logging.WARNING))

    hf_utils = None
    hf_was_disabled: bool | None = None
    try:
        import huggingface_hub.utils as hf_utils_module

        hf_utils = hf_utils_module
        are_disabled = getattr(hf_utils, "are_progress_bars_disabled", None)
        if callable(are_disabled):
            hf_was_disabled = bool(are_disabled())
    except Exception:  # noqa: BLE001 - optional dependency may not be installed yet.
        hf_utils = None

    transformers_logging = None
    transformers_progress_enabled: bool | None = None
    try:
        from transformers.utils import logging as transformers_logging_module

        transformers_logging = transformers_logging_module
        is_enabled = getattr(transformers_logging, "is_progress_bar_enabled", None)
        if callable(is_enabled):
            transformers_progress_enabled = bool(is_enabled())
    except Exception:  # noqa: BLE001 - optional dependency may not be installed yet.
        transformers_logging = None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"huggingface_hub(\..*)?")
            warnings.filterwarnings(
                "ignore",
                message=r".*HF_TOKEN.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*HF Hub.*",
            )
            if hf_utils is not None:
                disable_hf = getattr(hf_utils, "disable_progress_bars", None)
                if callable(disable_hf):
                    disable_hf()
            if transformers_logging is not None:
                disable_transformers = getattr(
                    transformers_logging, "disable_progress_bar", None
                )
                if callable(disable_transformers):
                    disable_transformers()
            yield
    finally:
        logging.disable(previous_logging_disable)
        if previous_hf_progress is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_hf_progress
        if previous_tokenizers_parallelism is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = previous_tokenizers_parallelism

        if hf_utils is not None and hf_was_disabled is False:
            enable_hf = getattr(hf_utils, "enable_progress_bars", None)
            if callable(enable_hf):
                enable_hf()
        if transformers_logging is not None and transformers_progress_enabled is True:
            enable_transformers = getattr(
                transformers_logging, "enable_progress_bar", None
            )
            if callable(enable_transformers):
                enable_transformers()


def build_embedder() -> EmbedderPort | None:
    """Build the configured embedder (bundled local default, or None to disable).

    ``CONTEXT_ENGINE_EMBEDDER``:
      - unset with no setup config / ``local`` / ``hashing`` → :class:`HashingEmbedder`
      - ``legacy`` / ``sentence-transformers`` → all-MiniLM-L6-v2 by default
      - ``auto`` → sentence-transformers when installed, otherwise hashing
      - ``none`` / ``off`` / ``lexical`` → ``None`` (labeled lexical fallback)

    ``CONTEXT_ENGINE_EMBEDDING_MODEL`` overrides the SentenceTransformer model
    name when the legacy embedder is selected.
    """
    choice = normalize_embedding_mode(configured_embedder_choice() or "local")
    if choice in DISABLED_EMBEDDER_ALIASES:
        return None
    if choice in HASHING_EMBEDDER_ALIASES:
        return HashingEmbedder()
    if choice in AUTO_SENTENCE_TRANSFORMER_ALIASES:
        if _sentence_transformers_installed():
            return _sentence_transformer_embedder()
        return HashingEmbedder()
    if choice in EXPLICIT_SENTENCE_TRANSFORMER_ALIASES:
        if not _sentence_transformers_installed():
            logger.warning(
                "sentence-transformers is not installed; using local hashing embedder"
            )
            return HashingEmbedder()
        return _sentence_transformer_embedder()
    # Unknown value: default to the bundled local embedder rather than crashing.
    return HashingEmbedder()


def configured_embedder_choice(*, include_env: bool = True) -> str | None:
    if include_env:
        raw = (os.getenv("CONTEXT_ENGINE_EMBEDDER") or "").strip()
        if raw:
            return raw
    config = _local_config()
    for key in ("embedder", "embedding_provider", "embedding_backend"):
        raw = config.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def configured_embedding_model(*, include_env: bool = True) -> str:
    if include_env:
        raw = (os.getenv("CONTEXT_ENGINE_EMBEDDING_MODEL") or "").strip()
        if raw:
            return raw
    config = _local_config()
    for key in ("embedding_model", "sentence_transformer_model"):
        raw = config.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return DEFAULT_SENTENCE_TRANSFORMER_MODEL


def default_sentence_transformer_cache(*, include_env: bool = True) -> str:
    if include_env:
        raw = (os.getenv("CONTEXT_ENGINE_EMBEDDING_CACHE") or "").strip()
        if raw:
            return raw
    config = _local_config()
    raw = config.get("embedding_cache")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return str(_default_home() / "models" / "sentence-transformers")


def _sentence_transformer_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        model_name=configured_embedding_model(),
        cache_folder=default_sentence_transformer_cache(),
    )


def _sentence_transformers_installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _configured_embedding_dimensions() -> int | None:
    raw = (os.getenv("CONTEXT_ENGINE_EMBEDDING_DIMENSIONS") or "").strip()
    if not raw:
        raw_value = _local_config().get("embedding_dimensions")
        raw = str(raw_value).strip() if raw_value is not None else ""
    if not raw:
        return None
    try:
        dim = int(raw)
    except ValueError:
        return None
    return dim if dim > 0 else None


def _known_sentence_transformer_dimensions(model_name: str) -> int | None:
    normalized = (model_name or "").strip().lower()
    return _KNOWN_SENTENCE_TRANSFORMER_DIMS.get(normalized)


def _local_config() -> dict[str, object]:
    try:
        with open(_default_home() / "config.json", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _default_home() -> Path:
    raw = (os.getenv("CONTEXT_ENGINE_HOME") or "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".potpie"


__all__ = [
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "HashingEmbedder",
    "SEMANTIC_EMBEDDER_ALIASES",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "configured_embedder_choice",
    "configured_embedding_model",
    "default_sentence_transformer_cache",
]
