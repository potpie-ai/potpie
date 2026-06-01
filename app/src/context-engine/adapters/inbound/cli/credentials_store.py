"""Persist Potpie API token for the Potpie CLI (user config dir)."""

from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Optional

import keyring
from keyring.errors import KeyringError

_CREDENTIALS_FILENAME = "credentials.json"
_CONFIG_DIR_NAME = "potpie"
_LEGACY_CONFIG_DIR_NAME = "context-engine"
_KEYRING_SERVICE = "potpie"
_GITHUB_TOKEN_SECRET = "github_token"


class CredentialStoreError(Exception):
    """Credential metadata or secure secret storage failure."""


# Alias used by GitHub CLI auth (matches integration branch naming).
ProviderCredentialError = CredentialStoreError


def config_dir() -> Path:
    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / _CONFIG_DIR_NAME
    return Path.home() / ".config" / _CONFIG_DIR_NAME


def legacy_config_dir() -> Path:
    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / _LEGACY_CONFIG_DIR_NAME
    return Path.home() / ".config" / _LEGACY_CONFIG_DIR_NAME


def credentials_path() -> Path:
    return config_dir() / _CREDENTIALS_FILENAME


def legacy_credentials_path() -> Path:
    return legacy_config_dir() / _CREDENTIALS_FILENAME


def readable_credentials_path() -> Path:
    path = credentials_path()
    if path.is_file():
        return path
    legacy = legacy_credentials_path()
    if legacy.is_file():
        return legacy
    return path


def read_credentials() -> dict[str, Any]:
    """Return parsed JSON or empty dict if missing/invalid."""
    path = readable_credentials_path()
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_payload(payload: dict[str, Any]) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = credentials_path()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _norm_secret_name(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        raise ValueError("secret name must be non-empty")
    return key


def store_secure_secret(name: str, secret: str, *, label: str | None = None) -> None:
    """Store a secret in the system keychain under the Potpie CLI service."""
    key = _norm_secret_name(name)
    try:
        keyring.set_password(_KEYRING_SERVICE, key, secret)
    except KeyringError as exc:
        raise CredentialStoreError(
            f"Failed to store {label or key} in system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise CredentialStoreError(
            f"Failed to store {label or key} in system keychain: {exc}"
        ) from exc


def load_secure_secret(name: str, *, label: str | None = None) -> str:
    """Load a secret from the system keychain, returning an empty string if absent."""
    key = _norm_secret_name(name)
    try:
        secret = keyring.get_password(_KEYRING_SERVICE, key)
    except KeyringError as exc:
        raise CredentialStoreError(
            f"Failed to read {label or key} from system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise CredentialStoreError(
            f"Failed to read {label or key} from system keychain: {exc}"
        ) from exc
    return secret or ""


def delete_secure_secret(name: str, *, label: str | None = None) -> None:
    """Remove a secret from the system keychain if the backend permits it."""
    key = _norm_secret_name(name)
    try:
        keyring.delete_password(_KEYRING_SERVICE, key)
    except KeyringError:
        pass
    except Exception as exc:
        raise CredentialStoreError(
            f"Failed to remove {label or key} from system keychain: {exc}"
        ) from exc


def _norm_integration_key(provider: str) -> str:
    key = str(provider or "").strip().lower()
    if not key:
        raise ValueError("integration provider must be non-empty")
    return key


def _read_integrations(payload: dict[str, Any]) -> dict[str, Any]:
    integrations = payload.get("integrations")
    return dict(integrations) if isinstance(integrations, dict) else {}


def get_integration_metadata(provider: str) -> dict[str, Any]:
    """Return non-secret integration metadata from credentials.json."""
    key = _norm_integration_key(provider)
    entry = _read_integrations(read_credentials()).get(key)
    return dict(entry) if isinstance(entry, dict) else {}


def write_integration_metadata(provider: str, metadata: dict[str, Any]) -> None:
    """Persist non-secret integration metadata under ``integrations.<provider>``."""
    key = _norm_integration_key(provider)
    payload: dict[str, Any] = dict(read_credentials())
    integrations = _read_integrations(payload)
    integrations[key] = dict(metadata)
    payload["integrations"] = integrations
    _write_payload(payload)


def clear_integration_metadata(provider: str) -> None:
    """Remove non-secret integration metadata for a provider."""
    key = _norm_integration_key(provider)
    payload: dict[str, Any] = dict(read_credentials())
    integrations = _read_integrations(payload)
    integrations.pop(key, None)
    if integrations:
        payload["integrations"] = integrations
    else:
        payload.pop("integrations", None)
    if not payload:
        path = credentials_path()
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.is_file():
                path.unlink()
        return
    _write_payload(payload)


def list_integration_metadata() -> dict[str, dict[str, Any]]:
    """Return all stored non-secret integration metadata keyed by provider."""
    integrations = _read_integrations(read_credentials())
    return {
        str(key): dict(value)
        for key, value in integrations.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def get_stored_api_key() -> str:
    v = read_credentials().get("api_key")
    return str(v).strip() if v else ""


def get_stored_api_base_url() -> str:
    v = read_credentials().get("api_base_url")
    if not v:
        return ""
    return str(v).strip().rstrip("/")


def write_credentials(*, api_key: str, api_base_url: Optional[str] = None) -> None:
    """Write credentials file with mode 0o600.

    Merges with existing file: if ``api_base_url`` is ``None``, any stored ``api_base_url`` is kept.
    Pass ``api_base_url=""`` to clear the stored base URL.
    """
    payload: dict[str, Any] = dict(read_credentials())
    payload["api_key"] = api_key.strip()
    if api_base_url is not None:
        u = api_base_url.strip().rstrip("/")
        if u:
            payload["api_base_url"] = u
        else:
            payload.pop("api_base_url", None)
    _write_payload(payload)


def clear_credentials() -> None:
    """Remove credentials file if it exists."""
    path = credentials_path()
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.is_file():
            path.unlink()


def clear_pot_scope_state() -> None:
    """Remove ``active_pot_id`` and ``pot_aliases``; keep API key / base URL if stored."""
    path = credentials_path()
    if not path.is_file():
        return
    payload: dict[str, Any] = dict(read_credentials())
    payload.pop("active_pot_id", None)
    payload.pop("pot_aliases", None)
    if not payload:
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.is_file():
                path.unlink()
        return
    _write_payload(payload)


def get_active_pot_id() -> str:
    """CLI-selected active pot (optional)."""
    v = read_credentials().get("active_pot_id")
    return str(v).strip() if v else ""


def set_active_pot_id(pot_id: str) -> None:
    """Persist active pot id alongside API credentials."""
    payload: dict[str, Any] = dict(read_credentials())
    payload["active_pot_id"] = pot_id.strip()
    _write_payload(payload)


def clear_active_pot_id() -> None:
    """Remove stored active pot id (fall back to env maps + git origin)."""
    path = credentials_path()
    if not path.is_file():
        return
    payload: dict[str, Any] = dict(read_credentials())
    payload.pop("active_pot_id", None)
    if not payload:
        path.unlink(missing_ok=True)
        return
    _write_payload(payload)


def _norm_alias_key(name: str) -> str:
    return name.strip().lower()


def get_pot_aliases() -> dict[str, str]:
    """Lowercase slug/local alias -> context pot UUID (from ``pot alias`` / ``pot create``)."""
    raw = read_credentials().get("pot_aliases")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            ks, vs = k.strip().lower(), v.strip()
            if ks and vs:
                out[ks] = vs
    return out


def register_pot_alias(name: str, pot_id: str) -> None:
    """Store a slug or friendly name -> context pot UUID (``pot alias``)."""
    key = _norm_alias_key(name)
    if not key:
        raise ValueError("Alias name must be non-empty")
    try:
        uid = uuid.UUID(str(pot_id).strip())
    except ValueError as e:
        raise ValueError("pot_id must be a UUID") from e
    payload: dict[str, Any] = dict(read_credentials())
    aliases = dict(payload.get("pot_aliases") or {})
    if not isinstance(aliases, dict):
        aliases = {}
    aliases[key] = str(uid)
    payload["pot_aliases"] = aliases
    _write_payload(payload)


def resolve_cli_pot_ref(ref: str) -> tuple[str | None, str]:
    """Resolve a pot argument to a canonical UUID string.

    Accepts a UUID or a name registered via ``potpie pot alias``.

    Returns ``(pot_id, "")`` on success, or ``(None, error_message)``.
    """
    s = (ref or "").strip()
    if not s:
        return None, "Pot reference is empty."
    try:
        u = uuid.UUID(s)
        return str(u), ""
    except ValueError:
        pass
    aliases = get_pot_aliases()
    k = _norm_alias_key(s)
    vs = aliases.get(k)
    if vs:
        try:
            u = uuid.UUID(vs.strip())
            return str(u), ""
        except ValueError:
            return None, f"Stored pot id for alias {s!r} is not a valid UUID."
    return None, (
        f"Unknown pot {s!r}. Run `potpie pot create <slug>` (server pot + alias), "
        f"or `potpie pot pots` for slugs/ids, then `pot use` / `pot alias`."
    )


def write_provider_credentials(provider: str, payload: dict[str, Any]) -> None:
    """Persist provider credentials (GitHub only on this branch)."""
    key = _norm_integration_key(provider)
    if key != "github":
        raise ValueError(f"Unsupported provider {provider!r}; expected 'github'.")
    stored_payload = dict(payload)
    access_token = str(stored_payload.pop("access_token", "") or "").strip()
    if not access_token:
        raise ProviderCredentialError("GitHub access token is required.")
    store_secure_secret(
        _GITHUB_TOKEN_SECRET,
        access_token,
        label="GitHub token",
    )
    stored_payload["token_storage"] = "keychain"
    write_integration_metadata(key, stored_payload)


def get_provider_credentials(provider: str) -> dict[str, Any]:
    """Return provider metadata merged with secrets from keychain."""
    key = _norm_integration_key(provider)
    if key != "github":
        return {}
    metadata = get_integration_metadata(key)
    if not metadata:
        return {}
    result = dict(metadata)
    token = load_secure_secret(_GITHUB_TOKEN_SECRET, label="GitHub token")
    if not token:
        raise ProviderCredentialError(
            "GitHub token not found in system keychain. Run: potpie auth github login"
        )
    result["access_token"] = token
    return result


def clear_provider_credentials(provider: str) -> None:
    """Remove provider secrets from keychain and drop integration metadata."""
    key = _norm_integration_key(provider)
    if key != "github":
        raise ValueError(f"Unsupported provider {provider!r}; expected 'github'.")
    delete_secure_secret(_GITHUB_TOKEN_SECRET, label="GitHub token")
    clear_integration_metadata(key)
