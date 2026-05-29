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
_GITHUB_TOKEN_USERNAME = "github_token"
_POTPIE_API_KEY_USERNAME = "potpie_api_key"
_POTPIE_FIREBASE_REFRESH_TOKEN_USERNAME = "potpie_firebase_refresh_token"
_POTPIE_FIREBASE_API_KEY_USERNAME = "potpie_firebase_api_key"


class ProviderCredentialError(Exception):
    """Provider credential storage or retrieval failure."""


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


def _store_github_token(access_token: str) -> None:
    try:
        keyring.set_password(_KEYRING_SERVICE, _GITHUB_TOKEN_USERNAME, access_token)
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to store GitHub token in system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to store GitHub token in system keychain: {exc}"
        ) from exc


def _load_github_token() -> str:
    try:
        access_token = keyring.get_password(_KEYRING_SERVICE, _GITHUB_TOKEN_USERNAME)
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to read GitHub token from system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to read GitHub token from system keychain: {exc}"
        ) from exc
    if access_token is None:
        raise ProviderCredentialError(
            "GitHub token not found in system keychain. Run: potpie github login"
        )
    return access_token


def _delete_github_token() -> None:
    try:
        keyring.delete_password(_KEYRING_SERVICE, _GITHUB_TOKEN_USERNAME)
    except KeyringError:
        pass
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to remove GitHub token from system keychain: {exc}"
        ) from exc


def _store_potpie_api_key(api_key: str) -> None:
    try:
        keyring.set_password(_KEYRING_SERVICE, _POTPIE_API_KEY_USERNAME, api_key)
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to store Potpie API key in system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to store Potpie API key in system keychain: {exc}"
        ) from exc


def _load_potpie_api_key() -> str | None:
    try:
        return keyring.get_password(_KEYRING_SERVICE, _POTPIE_API_KEY_USERNAME)
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to read Potpie API key from system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to read Potpie API key from system keychain: {exc}"
        ) from exc


def _delete_potpie_api_key() -> None:
    try:
        keyring.delete_password(_KEYRING_SERVICE, _POTPIE_API_KEY_USERNAME)
    except KeyringError:
        pass
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to remove Potpie API key from system keychain: {exc}"
        ) from exc


def _store_potpie_firebase_refresh_token(refresh_token: str) -> None:
    try:
        keyring.set_password(
            _KEYRING_SERVICE,
            _POTPIE_FIREBASE_REFRESH_TOKEN_USERNAME,
            refresh_token,
        )
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to store Potpie refresh token in system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to store Potpie refresh token in system keychain: {exc}"
        ) from exc


def _store_potpie_firebase_api_key(firebase_api_key: str) -> None:
    try:
        keyring.set_password(
            _KEYRING_SERVICE,
            _POTPIE_FIREBASE_API_KEY_USERNAME,
            firebase_api_key,
        )
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to store Potpie Firebase API key in system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to store Potpie Firebase API key in system keychain: {exc}"
        ) from exc


def get_potpie_firebase_refresh_token() -> str:
    try:
        token = keyring.get_password(
            _KEYRING_SERVICE,
            _POTPIE_FIREBASE_REFRESH_TOKEN_USERNAME,
        )
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to read Potpie refresh token from system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to read Potpie refresh token from system keychain: {exc}"
        ) from exc
    return str(token or "").strip()


def _load_potpie_firebase_api_key() -> str | None:
    try:
        return keyring.get_password(
            _KEYRING_SERVICE,
            _POTPIE_FIREBASE_API_KEY_USERNAME,
        )
    except KeyringError as exc:
        raise ProviderCredentialError(
            f"Failed to read Potpie Firebase API key from system keychain: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to read Potpie Firebase API key from system keychain: {exc}"
        ) from exc


def _delete_potpie_firebase_refresh_token() -> None:
    try:
        keyring.delete_password(_KEYRING_SERVICE, _POTPIE_FIREBASE_REFRESH_TOKEN_USERNAME)
    except KeyringError:
        pass
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to remove Potpie refresh token from system keychain: {exc}"
        ) from exc


def _delete_potpie_firebase_api_key() -> None:
    try:
        keyring.delete_password(_KEYRING_SERVICE, _POTPIE_FIREBASE_API_KEY_USERNAME)
    except KeyringError:
        pass
    except Exception as exc:
        raise ProviderCredentialError(
            f"Failed to remove Potpie Firebase API key from system keychain: {exc}"
        ) from exc


def get_stored_api_key() -> str:
    try:
        from_keychain = _load_potpie_api_key()
        if from_keychain and from_keychain.strip():
            return from_keychain.strip()
    except ProviderCredentialError:
        pass
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


def write_potpie_auth_metadata(
    *,
    created_at: str,
    auth_type: str = "api_key",
) -> None:
    """Persist Potpie CLI auth metadata only (secrets stay in keychain)."""
    payload: dict[str, Any] = dict(read_credentials())
    integrations = dict(payload.get("integrations") or {})
    if not isinstance(integrations, dict):
        integrations = {}
    potpie_metadata = {
        "auth_type": auth_type,
        "token_storage": "keychain",
        "created_at": created_at,
    }
    integrations["potpie"] = potpie_metadata
    payload["integrations"] = integrations
    _write_payload(payload)


def clear_potpie_auth(*, clear_api_key: bool = False) -> None:
    """Remove Potpie auth secrets from keychain and drop integrations.potpie metadata."""
    if clear_api_key:
        _delete_potpie_api_key()
    _delete_potpie_firebase_refresh_token()
    _delete_potpie_firebase_api_key()
    payload: dict[str, Any] = dict(read_credentials())
    integrations = dict(payload.get("integrations") or {})
    if isinstance(integrations, dict):
        integrations.pop("potpie", None)
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


def store_potpie_api_key(api_key: str, *, created_at: str) -> None:
    """Store Potpie API key in keychain and write metadata to credentials.json."""
    key = api_key.strip()
    if not key.startswith("sk-"):
        raise ProviderCredentialError("Potpie API key must start with sk-.")
    _store_potpie_api_key(key)
    write_potpie_auth_metadata(created_at=created_at, auth_type="api_key")


def store_potpie_firebase_refresh_token(
    refresh_token: str,
    *,
    created_at: str,
    firebase_api_key: str | None = None,
) -> None:
    """Store Firebase refresh token in keychain and write metadata only."""
    token = refresh_token.strip()
    if not token:
        raise ProviderCredentialError("Potpie Firebase refresh token is required.")
    _store_potpie_firebase_refresh_token(token)
    if firebase_api_key and firebase_api_key.strip():
        _store_potpie_firebase_api_key(firebase_api_key.strip())
    write_potpie_auth_metadata(
        created_at=created_at,
        auth_type="firebase_session",
    )


def update_potpie_firebase_refresh_token(refresh_token: str) -> None:
    """Update rotated Firebase refresh token without touching metadata."""
    token = refresh_token.strip()
    if not token:
        raise ProviderCredentialError("Potpie Firebase refresh token is required.")
    _store_potpie_firebase_refresh_token(token)


def get_potpie_auth_type() -> str:
    integrations = read_credentials().get("integrations")
    if not isinstance(integrations, dict):
        return ""
    payload = integrations.get("potpie")
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("auth_type") or "").strip()


def get_potpie_firebase_api_key() -> str:
    try:
        from_keychain = _load_potpie_firebase_api_key()
        if from_keychain and from_keychain.strip():
            return from_keychain.strip()
    except ProviderCredentialError:
        pass
    integrations = read_credentials().get("integrations")
    if not isinstance(integrations, dict):
        return ""
    payload = integrations.get("potpie")
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("firebase_api_key") or "").strip()


def clear_credentials() -> None:
    """Remove credentials file if it exists."""
    _delete_potpie_api_key()
    _delete_potpie_firebase_refresh_token()
    _delete_potpie_firebase_api_key()
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


def write_provider_credentials(provider: str, payload: dict[str, Any]) -> None:
    key = str(provider or "").strip().lower()
    if not key:
        raise ValueError("provider must be non-empty")
    stored_payload = dict(payload)
    if key == "github":
        access_token = str(stored_payload.pop("access_token", "") or "").strip()
        if not access_token:
            raise ProviderCredentialError("GitHub access token is required.")
        _store_github_token(access_token)
        stored_payload["token_storage"] = "keychain"
    existing: dict[str, Any] = dict(read_credentials())
    integrations = dict(existing.get("integrations") or {})
    if not isinstance(integrations, dict):
        integrations = {}
    integrations[key] = stored_payload
    existing["integrations"] = integrations
    _write_payload(existing)


def get_provider_credentials(provider: str) -> dict[str, Any]:
    key = str(provider or "").strip().lower()
    if not key:
        return {}
    integrations = read_credentials().get("integrations")
    if not isinstance(integrations, dict):
        return {}
    payload = integrations.get(key)
    if not isinstance(payload, dict):
        return {}
    result = dict(payload)
    if key == "github":
        result["access_token"] = _load_github_token()
    return result


def clear_provider_credentials(provider: str) -> None:
    """Remove provider secrets from keychain and drop integrations metadata."""
    key = str(provider or "").strip().lower()
    if not key:
        raise ValueError("provider must be non-empty")
    if key == "github":
        _delete_github_token()
    payload: dict[str, Any] = dict(read_credentials())
    integrations = dict(payload.get("integrations") or {})
    if isinstance(integrations, dict):
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
