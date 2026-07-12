"""Root runtime wrapper for context-engine project ``.env`` loading."""

from __future__ import annotations

from collections.abc import Collection

from potpie_context_engine.bootstrap import env_bootstrap as _engine_env_bootstrap

_PROTECTED_DOTENV_KEYS = _engine_env_bootstrap._PROTECTED_DOTENV_KEYS
_PROJECT_ROOT_MARKERS = _engine_env_bootstrap._PROJECT_ROOT_MARKERS
_parse_env_line = _engine_env_bootstrap._parse_env_line
_load_env_file = _engine_env_bootstrap._load_env_file
_load_monorepo_potpie_env = _engine_env_bootstrap._load_monorepo_potpie_env


def load_cli_env(*, skip_keys: Collection[str] = _PROTECTED_DOTENV_KEYS) -> None:
    """Merge the trusted project ``.env`` using the context-engine implementation."""
    _engine_env_bootstrap.load_cli_env(skip_keys=skip_keys)


__all__ = ["load_cli_env"]
