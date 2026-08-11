"""``LocalConfigService`` — local home dir + JSON config file.

Backs the first setup step. State lives at ``<home>/config.json`` where
``<home>`` is ``$CONTEXT_ENGINE_HOME`` or ``~/.potpie`` (shared with
:func:`potpie_context_engine.adapters.outbound.pots.local_pot_store.default_home`). This is a working
Real dirs + JSON, not a stub — config is cheap and unblocks every
downstream step. The real config layer may add schema/validation behind the same
``ConfigService`` interface.
"""

from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_core.lifecycle import SetupPlan

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

# Keys the runtime still honours but never advertises. ``configured_embedder_choice``
# / ``configured_embedding_model`` (local_embedder.py) fall back to these older
# spellings after the catalog names, so a writer that only accepted
# ``KNOWN_CONFIG_KEYS`` would refuse a key the reader demonstrably obeys — the CLI
# lying in the other direction. They stay out of the advertised catalog because
# `config list`'s ``known_keys`` and the sub-app help are how a user learns the
# *current* names; drop an entry here only once nothing reads it.
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

    The catalog was decorative — printed in the sub-app help and echoed as
    ``known_keys`` by ``config list``, but never enforced — so ``config set``
    persisted ``emebdder`` and reported success for a setting nothing would ever
    look at. Callers that gate writes on this turn that silent no-op into a
    refusal, and keep ``config.json`` (written at the process umask, unlike every
    credential store in this repo) from being used as an arbitrary secret store.
    """
    return key in KNOWN_CONFIG_KEYS or key in _ACCEPTED_ALIAS_KEYS


def _without_url_userinfo(value: str) -> str:
    """Blank the ``user:password@`` an operator may have typed inside a URL.

    :func:`is_secret_config_key` classifies by *key name*, which by construction
    cannot see a credential that arrived in the value. ``ledger.url`` is not a
    secret-shaped key, so ``config set ledger.url https://user:tok@host`` echoed
    the token straight back to stdout and into the ``--json`` payload — the same
    leak the key-based redaction closes one column to the left, through the one
    kind of value in this catalog that routinely carries a credential.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        # Not parseable as a URL: nothing here can claim to know where a
        # credential would be, so changing the value would be guessing.
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

    def unset(self, key: str) -> bool:
        """Drop ``key`` from the file; report whether it was actually there.

        Unlike :meth:`set`, this accepts keys outside the catalog on purpose.
        The write gate strands every key ``set`` used to accept, and the ones
        worth caring about are the credentials that got in while it accepted
        anything — so the one command that can clear them must not consult the
        same catalog that refuses to rewrite them.

        Returning a bool rather than raising keeps the caller honest: "there was
        nothing to remove" is a different sentence from "removed", and reporting
        the second for the first is the silent-success shape this CLI has been
        pulling out of its commands.
        """
        data = self._load()
        if key not in data:
            return False
        del data[key]
        self.ensure_home()
        self._save(data)
        return True

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
        # Owner-only, and set on the temp *before* the rename so the file is never
        # briefly world-readable — the ordering credentials_store/hosts/ipc_auth
        # all use. Nothing in the catalog is a credential, but `config set` used
        # to accept any key at all, so real homes out there already have tokens
        # in this file; it was the only state file in the repo left at umask.
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)
        tmp.replace(self._path)
        self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)


__all__ = [
    "KNOWN_CONFIG_KEYS",
    "LocalConfigService",
    "is_known_config_key",
    "is_secret_config_key",
    "public_config_value",
]
