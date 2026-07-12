"""``LocalConfigService`` — local home dir + JSON config file.

Backs the first setup step. State lives at ``<home>/config.json`` where
``<home>`` is ``$CONTEXT_ENGINE_HOME`` or ``~/.potpie`` (shared with
:func:`adapters.outbound.pots.local_pot_store.default_home`). This is a working
Real dirs + JSON, not a stub — config is cheap and unblocks every
downstream step. The real config layer may add schema/validation behind the same
``ConfigService`` interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.domain.lifecycle import SetupPlan

KNOWN_CONFIG_KEYS: tuple[str, ...] = (
    "profile",
    "backend",
    "home",
    "embedder",
    "embedding_model",
    "embedding_cache",
    "ledger.binding",
    "ledger.org",
    "ledger.url",
)

_SECRET_KEY_MARKERS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "api_key",
    "api-key",
    "credential",
)

_REDACTED = "<redacted>"

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATOR_RE = re.compile(r"[_\-.\s]+")


def _segment_key_words(text: str) -> list[str]:
    """Break a config key into lowercase word segments.

    Splits on separators (``_ - . space``) and camelCase boundaries so the
    matcher can compare whole words instead of raw substrings (avoids false
    positives like ``max_tokens`` or ``tokenizer``).
    """
    spaced = _SEPARATOR_RE.sub(" ", text)
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", spaced)
    return [word for word in spaced.lower().split() if word]


_MARKER_WORD_SEQUENCES: tuple[tuple[str, ...], ...] = tuple(
    dict.fromkeys(
        tuple(_segment_key_words(marker))
        for marker in _SECRET_KEY_MARKERS
        if _segment_key_words(marker)
    )
)

# Single-word markers (e.g. ``token``) match on whole-word boundaries so
# ``tokenizer``/``max_tokens`` are not false positives. Compound markers
# (e.g. ``api_key`` → ``apikey``) match the joined key so separator-less
# variants like ``apikey`` are still caught.
_SINGLE_WORD_SECRET_MARKERS: frozenset[str] = frozenset(
    seq[0] for seq in _MARKER_WORD_SEQUENCES if len(seq) == 1
)
_COMPOUND_SECRET_MARKERS: tuple[str, ...] = tuple(
    dict.fromkeys("".join(seq) for seq in _MARKER_WORD_SEQUENCES if len(seq) > 1)
)


def is_secret_config_key(key: str) -> bool:
    words = _segment_key_words(key)
    if any(word in _SINGLE_WORD_SECRET_MARKERS for word in words):
        return True
    joined = "".join(words)
    return any(compound in joined for compound in _COMPOUND_SECRET_MARKERS)


def public_config_value(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if is_secret_config_key(key):
        return _REDACTED
    return str(value)


@dataclass(slots=True)
class LocalConfigService:
    """Flat-file config provisioning + get/set."""

    home: Path = field(default_factory=default_home)

    @property
    def _path(self) -> Path:
        return self.home / "config.json"

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        return self.home

    def write_defaults(self, plan: SetupPlan) -> Path:
        self.ensure_home()
        data = self._load()
        # Only fill values the user has not already set (idempotent re-runs).
        data.setdefault("profile", plan.mode)
        data.setdefault("backend", plan.backend)
        data.setdefault("home", str(self.home))
        data.setdefault("embedder", plan.embeddings)
        data.setdefault("embedding_model", plan.embedding_model)
        data.setdefault(
            "embedding_cache",
            str(self.home / "models" / "sentence-transformers"),
        )
        self._save(data)
        return self._path

    def get(self, key: str) -> str | None:
        value = self._load().get(key)
        return None if value is None else str(value)

    def list_public(self) -> dict[str, str | None]:
        """Return all config entries with secret-like keys redacted."""
        return {
            key: public_config_value(key, value)
            for key, value in sorted(self._load().items())
        }

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = value
        self.ensure_home()
        self._save(data)

    def probe(self) -> dict[str, Any]:
        return {"home": str(self.home), "config_exists": self._path.exists()}

    # --- raw state ----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self._path)


__all__ = [
    "KNOWN_CONFIG_KEYS",
    "LocalConfigService",
    "is_secret_config_key",
    "public_config_value",
]
