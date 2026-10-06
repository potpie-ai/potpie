"""``LocalConfigService`` — local home dir + JSON config file.

Backs the first setup step. State lives at ``<home>/config.json`` where
``<home>`` is ``$CONTEXT_ENGINE_HOME`` or ``~/.potpie`` (shared with
:func:`potpie.config.local_paths.default_home`). This is a working
Real dirs + JSON, not a stub — config is cheap and unblocks every
downstream step. The real config layer may add schema/validation behind the same
``ConfigService`` interface.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from potpie.config.local_paths import default_home
from potpie.config.local_state import local_json_transaction
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    embedding_cache_override,
)
from potpie_context_engine.core.lifecycle import SetupPlan
from potpie_context_engine.domain.embedding_modes import (
    EMBEDDING_MODEL_PREP_SKIPPED_ALIASES,
    normalize_embedding_mode,
)

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

# Keys the runtime still honours but never advertises: the embedder settings
# fall back to these older spellings after the catalog names, so a writer that
# accepted only ``KNOWN_CONFIG_KEYS`` would refuse a key the reader obeys. They
# stay out of the advertised catalog (``config list`` ``known_keys`` and the
# sub-app help teach the current names); drop one only once nothing reads it.
_ACCEPTED_ALIAS_KEYS: tuple[str, ...] = (
    "embedding_provider",
    "embedding_backend",
    "sentence_transformer_model",
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


def is_known_config_key(key: str) -> bool:
    """Is this a key some part of the system actually reads?

    ``config set`` gates writes on this, so a typo (``emebdder``) is refused
    instead of persisted and reported as set, and ``config.json`` cannot be
    used as an arbitrary secret store.
    """
    return key in KNOWN_CONFIG_KEYS or key in _ACCEPTED_ALIAS_KEYS


def _without_url_userinfo(value: str) -> str:
    """Blank the ``user:password@`` an operator may have typed inside a URL.

    :func:`is_secret_config_key` classifies by key name, which cannot see a
    credential that arrived in the value: ``ledger.url`` is not a secret-shaped
    key, so ``https://user:tok@host`` would otherwise be echoed back verbatim.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        # Not parseable as a URL: changing the value would be guessing.
        return value
    if not parts.scheme or "@" not in parts.netloc:
        return value
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"{_REDACTED}@{host}"))


def public_config_value(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if is_secret_config_key(key):
        return _REDACTED
    return _without_url_userinfo(str(value))


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
        with local_json_transaction(self._path, default_factory=dict) as data:
            # Only fill values the user has not already set (idempotent re-runs).
            data.setdefault("profile", plan.mode)
            data.setdefault("backend", plan.backend)
            data.setdefault("home", str(self.home))
            data.setdefault("embedder", plan.embeddings)
            data.setdefault("embedding_model", plan.embedding_model)
            cache = _embedding_cache_location(plan, home=self.home)
            if cache is not None:
                data.setdefault("embedding_cache", cache)
        _owner_only(self._path)
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
        with local_json_transaction(self._path, default_factory=dict) as data:
            data[key] = value
        _owner_only(self._path)

    def unset(self, key: str) -> bool:
        """Drop ``key`` from the file; report whether it was there.

        Unlike :meth:`set`, this accepts keys outside the catalog on purpose:
        the write gate strands every key ``set`` used to accept (credentials
        among them), and removal is the only repair left for those.
        """
        if key not in self._load():
            return False
        removed = False
        with local_json_transaction(self._path, default_factory=dict) as data:
            if key in data:
                del data[key]
                removed = True
        _owner_only(self._path)
        return removed

    def probe(self) -> dict[str, Any]:
        return {"home": str(self.home), "config_exists": self._path.exists()}

    # --- raw state ----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def _owner_only(path: Path) -> None:
    """Keep ``config.json`` readable by its owner only (0600).

    The transactional writer already replaces the file with an owner-only
    temporary; this also tightens a file an older release left at the umask,
    since ``config set`` once accepted any key, credentials included.
    """
    if os.name == "nt":
        return
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except FileNotFoundError:
        return


def _embedding_cache_location(plan: SetupPlan, *, home: Path) -> str | None:
    """Where this plan's model weights will actually land, or ``None``.

    ``None`` for the modes that cache nothing (no embedder, or the bundled
    hashing embedder). Otherwise the environment override wins, because that is
    what the runtime reads ahead of this file.
    """
    if (
        normalize_embedding_mode(plan.embeddings)
        in EMBEDDING_MODEL_PREP_SKIPPED_ALIASES
    ):
        return None
    return embedding_cache_override() or str(home / "models" / "sentence-transformers")


__all__ = [
    "KNOWN_CONFIG_KEYS",
    "LocalConfigService",
    "is_known_config_key",
    "is_secret_config_key",
    "public_config_value",
]
