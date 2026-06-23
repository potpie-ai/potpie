"""Persist Potpie API token for the Potpie CLI (user config dir)."""

from __future__ import annotations

import json
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import keyring
from keyring.errors import KeyringError

from context_engine.adapters.outbound.cli_auth.errors import CliAuthError

_CREDENTIALS_FILENAME = "credentials.json"
_INTEGRATION_SECRETS_FILENAME = "integration_secrets.json"
_CONFIG_DIR_NAME = "potpie"
_LEGACY_CONFIG_DIR_NAME = "context-engine"
_KEYRING_SERVICE = "potpie"
_POTPIE_API_KEY_SECRET = "potpie_api_key"
_POTPIE_FIREBASE_ID_TOKEN_SECRET = "potpie_firebase_id_token"
_POTPIE_FIREBASE_REFRESH_TOKEN_SECRET = "potpie_firebase_refresh_token"
_POTPIE_FIREBASE_API_KEY_SECRET = "potpie_firebase_api_key"
_GITHUB_TOKEN_SECRET = "github_token"
_LINEAR_ACCESS_TOKEN_SECRET = "linear_access_token"
_LINEAR_REFRESH_TOKEN_SECRET = "linear_refresh_token"
_LINEAR_CREDENTIALS_KEY = "linear"
_ATLASSIAN_LEGACY_TOKEN_SECRET = "atlassian_api_token"
_JIRA_TOKEN_SECRET = "jira_api_token"
_CONFLUENCE_TOKEN_SECRET = "confluence_api_token"
_ATLASSIAN_CREDENTIALS_KEY = "atlassian"
_JIRA_CREDENTIALS_KEY = "jira"
_CONFLUENCE_CREDENTIALS_KEY = "confluence"


class CredentialStoreError(CliAuthError):
    """Credential metadata or secure secret storage failure."""


class ProviderCredentialError(CredentialStoreError):
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


def integration_secrets_path() -> Path:
    """Linux integration token store (GitHub, Linear, Jira, Confluence)."""
    return config_dir() / _INTEGRATION_SECRETS_FILENAME


def integration_token_storage() -> str:
    """Return metadata token_storage value for CLI integrations on this platform."""
    return "file" if sys.platform == "linux" else "keychain"


def _storage_label(token_storage: str | None) -> str:
    return "local credentials file" if token_storage == "file" else "system keychain"


def _integration_secret_store_label() -> str:
    return _storage_label(integration_token_storage())


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
    _write_payload_to_path(path, payload)


def _write_payload_to_path(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _norm_secret_name(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        raise ValueError("secret name must be non-empty")
    return key


def _is_integration_secret_name(name: str) -> bool:
    """Secret keys used by GitHub, Linear, Jira, and Confluence CLI integrations."""
    key = _norm_secret_name(name)
    if key in {
        _GITHUB_TOKEN_SECRET,
        _LINEAR_ACCESS_TOKEN_SECRET,
        _LINEAR_REFRESH_TOKEN_SECRET,
        _JIRA_TOKEN_SECRET,
        _CONFLUENCE_TOKEN_SECRET,
        _ATLASSIAN_LEGACY_TOKEN_SECRET,
    }:
        return True
    return key.startswith("linear_access_token_") or key.startswith(
        "linear_refresh_token_"
    )


def _uses_linux_integration_file_storage(name: str) -> bool:
    return sys.platform == "linux" and _is_integration_secret_name(name)


def _read_integration_secrets_file() -> dict[str, str]:
    path = integration_secrets_path()
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and value is not None:
            out[key] = str(value)
    return out


def _write_integration_secrets_file(secrets: dict[str, str]) -> None:
    path = integration_secrets_path()
    if not secrets:
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.is_file():
                path.unlink()
        return
    _write_payload_to_path(path, secrets)


def _store_integration_file_secret(
    name: str,
    secret: str,
    *,
    label: str | None = None,
) -> None:
    key = _norm_secret_name(name)
    secrets = _read_integration_secrets_file()
    secrets[key] = secret
    try:
        _write_integration_secrets_file(secrets)
    except OSError as exc:
        raise CredentialStoreError(
            f"Failed to store {label or key} in local credentials file: {exc}"
        ) from exc


def _load_integration_file_secret(name: str) -> str:
    key = _norm_secret_name(name)
    return _read_integration_secrets_file().get(key, "")


def _delete_integration_file_secret(name: str) -> None:
    key = _norm_secret_name(name)
    secrets = _read_integration_secrets_file()
    if key not in secrets:
        return
    secrets.pop(key, None)
    try:
        _write_integration_secrets_file(secrets)
    except OSError as exc:
        raise CredentialStoreError(
            f"Failed to remove {key} from local credentials file: {exc}"
        ) from exc


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
    try:
        from_keychain = load_secure_secret(
            _POTPIE_API_KEY_SECRET,
            label="Potpie API key",
        )
        if from_keychain.strip():
            return from_keychain.strip()
    except CredentialStoreError:
        pass
    v = read_credentials().get("api_key")
    return str(v).strip() if v else ""


def get_stored_api_base_url() -> str:
    v = read_credentials().get("api_base_url")
    if not v:
        return ""
    return str(v).strip().rstrip("/")


def write_api_base_url(api_base_url: Optional[str]) -> None:
    """Persist non-secret Potpie API base URL without touching stored secrets."""
    payload: dict[str, Any] = dict(read_credentials())
    if api_base_url is not None:
        u = api_base_url.strip().rstrip("/")
        if u:
            payload["api_base_url"] = u
        else:
            payload.pop("api_base_url", None)
    _write_payload(payload)


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
    """Persist Potpie CLI auth metadata only; secrets stay in secure storage."""
    write_integration_metadata(
        "potpie",
        {
            "auth_type": auth_type,
            "token_storage": "keychain",
            "created_at": created_at,
        },
    )


def store_potpie_api_key(api_key: str, *, created_at: str) -> None:
    """Store Potpie API key in secure storage and write non-secret metadata."""
    key = api_key.strip()
    if not key.startswith("sk-"):
        raise CredentialStoreError("Potpie API key must start with sk-.")
    store_secure_secret(_POTPIE_API_KEY_SECRET, key, label="Potpie API key")
    write_potpie_auth_metadata(created_at=created_at, auth_type="api_key")


def store_potpie_firebase_refresh_token(
    refresh_token: str,
    *,
    created_at: str,
    firebase_api_key: str | None = None,
) -> None:
    """Store Firebase refresh token in secure storage and write metadata only."""
    token = refresh_token.strip()
    if not token:
        raise CredentialStoreError("Potpie Firebase refresh token is required.")
    store_secure_secret(
        _POTPIE_FIREBASE_REFRESH_TOKEN_SECRET,
        token,
        label="Potpie refresh token",
    )
    if firebase_api_key and firebase_api_key.strip():
        store_secure_secret(
            _POTPIE_FIREBASE_API_KEY_SECRET,
            firebase_api_key.strip(),
            label="Potpie Firebase API key",
        )
    write_potpie_auth_metadata(
        created_at=created_at,
        auth_type="potpie",
    )


def store_potpie_firebase_id_token(id_token: str) -> None:
    """Store Firebase ID token in secure storage."""
    token = id_token.strip()
    if not token:
        raise CredentialStoreError("Potpie Firebase ID token is required.")
    store_secure_secret(
        _POTPIE_FIREBASE_ID_TOKEN_SECRET,
        token,
        label="Potpie Firebase ID token",
    )


def update_potpie_firebase_refresh_token(refresh_token: str) -> None:
    """Update rotated Firebase refresh token without touching metadata."""
    token = refresh_token.strip()
    if not token:
        raise CredentialStoreError("Potpie Firebase refresh token is required.")
    store_secure_secret(
        _POTPIE_FIREBASE_REFRESH_TOKEN_SECRET,
        token,
        label="Potpie refresh token",
    )


def get_potpie_auth_type() -> str:
    metadata = get_integration_metadata("potpie")
    return str(metadata.get("auth_type") or "").strip()


def get_potpie_firebase_refresh_token() -> str:
    return load_secure_secret(
        _POTPIE_FIREBASE_REFRESH_TOKEN_SECRET,
        label="Potpie refresh token",
    ).strip()


def get_potpie_firebase_id_token() -> str:
    return load_secure_secret(
        _POTPIE_FIREBASE_ID_TOKEN_SECRET,
        label="Potpie Firebase ID token",
    ).strip()


def get_potpie_firebase_api_key() -> str:
    from_keychain = load_secure_secret(
        _POTPIE_FIREBASE_API_KEY_SECRET,
        label="Potpie Firebase API key",
    )
    if from_keychain.strip():
        return from_keychain.strip()
    metadata = get_integration_metadata("potpie")
    return str(metadata.get("firebase_api_key") or "").strip()


def clear_potpie_auth(*, clear_api_key: bool = False) -> None:
    """Remove Potpie session secrets and drop integrations.potpie metadata."""
    if clear_api_key:
        delete_secure_secret(_POTPIE_API_KEY_SECRET, label="Potpie API key")
    path = readable_credentials_path()
    if path.is_file():
        payload: dict[str, Any] = dict(read_credentials())
        removed = payload.pop("api_key", None) is not None
        if removed:
            if payload:
                _write_payload_to_path(path, payload)
            else:
                try:
                    path.unlink(missing_ok=True)
                except TypeError:
                    if path.is_file():
                        path.unlink()
    delete_secure_secret(
        _POTPIE_FIREBASE_ID_TOKEN_SECRET,
        label="Potpie Firebase ID token",
    )
    delete_secure_secret(
        _POTPIE_FIREBASE_REFRESH_TOKEN_SECRET,
        label="Potpie refresh token",
    )
    delete_secure_secret(
        _POTPIE_FIREBASE_API_KEY_SECRET,
        label="Potpie Firebase API key",
    )
    clear_integration_metadata("potpie")


def clear_credentials() -> None:
    """Remove credentials file if it exists."""
    delete_secure_secret(_POTPIE_API_KEY_SECRET, label="Potpie API key")
    delete_secure_secret(
        _POTPIE_FIREBASE_ID_TOKEN_SECRET,
        label="Potpie Firebase ID token",
    )
    delete_secure_secret(
        _POTPIE_FIREBASE_REFRESH_TOKEN_SECRET,
        label="Potpie refresh token",
    )
    delete_secure_secret(
        _POTPIE_FIREBASE_API_KEY_SECRET,
        label="Potpie Firebase API key",
    )
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


def _get_secret_or_empty(name: str, *, label: str) -> str:
    if _uses_linux_integration_file_storage(name):
        value = _load_integration_file_secret(name)
        if value:
            return value
        try:
            legacy = load_secure_secret(name, label=label)
        except CredentialStoreError as exc:
            raise ProviderCredentialError(str(exc)) from exc
        return legacy
    try:
        return load_secure_secret(name, label=label)
    except CredentialStoreError as exc:
        raise ProviderCredentialError(str(exc)) from exc


def _store_secret(name: str, secret: str, *, label: str) -> str:
    if _uses_linux_integration_file_storage(name):
        try:
            _store_integration_file_secret(name, secret, label=label)
        except CredentialStoreError as exc:
            raise ProviderCredentialError(str(exc)) from exc
        return "file"
    try:
        store_secure_secret(name, secret, label=label)
        return "keychain"
    except CredentialStoreError as exc:
        raise ProviderCredentialError(str(exc)) from exc


def _delete_secret(name: str, *, label: str) -> None:
    if _uses_linux_integration_file_storage(name):
        try:
            _delete_integration_file_secret(name)
        except CredentialStoreError as exc:
            raise ProviderCredentialError(str(exc)) from exc
        try:
            delete_secure_secret(name, label=label)
        except CredentialStoreError as exc:
            raise ProviderCredentialError(str(exc)) from exc
        return
    try:
        delete_secure_secret(name, label=label)
    except CredentialStoreError as exc:
        raise ProviderCredentialError(str(exc)) from exc


def _store_keychain_secret(label: str, username: str, secret: str) -> str:
    return _store_secret(username, secret, label=label)


def _load_keychain_secret(label: str, username: str) -> str:
    return _get_secret_or_empty(username, label=label)


def _delete_keychain_secret(label: str, username: str) -> None:
    _delete_secret(username, label=label)


def _read_metadata_entry(key: str) -> dict[str, Any]:
    return get_integration_metadata(key)


def _write_metadata_entry(key: str, metadata: dict[str, Any]) -> None:
    write_integration_metadata(key, metadata)


def _clear_metadata_entries(*keys: str) -> None:
    for key in keys:
        clear_integration_metadata(key)


def _linear_access_token_secret(org_id: str) -> str:
    return f"linear_access_token_{org_id}"


def _linear_refresh_token_secret(org_id: str) -> str:
    return f"linear_refresh_token_{org_id}"


def _normalize_linear_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Ensure multi-workspace ``organizations`` map exists; migrate legacy shape."""
    if not metadata:
        return {}
    meta = dict(metadata)
    orgs = meta.get("organizations")
    if isinstance(orgs, dict) and orgs:
        if not meta.get("active_organization_id"):
            meta["active_organization_id"] = next(iter(orgs.keys()))
        return meta

    org = meta.get("organization")
    if not isinstance(org, dict) or not org.get("id"):
        return meta

    org_id = str(org["id"])
    org_entry: dict[str, Any] = {
        "id": org_id,
        "name": org.get("name"),
        "url_key": org.get("url_key") or org.get("urlKey"),
        "account": meta.get("account"),
        "expires_at": meta.get("expires_at"),
        "scopes": meta.get("scopes") or meta.get("scope"),
        "stored_at": meta.get("stored_at"),
    }
    meta["organizations"] = {org_id: org_entry}
    meta["active_organization_id"] = org_id
    return meta


def _read_linear_metadata() -> dict[str, Any]:
    return _normalize_linear_metadata(_read_metadata_entry(_LINEAR_CREDENTIALS_KEY) or {})


def get_active_linear_organization_id() -> str | None:
    meta = _read_linear_metadata()
    active = str(meta.get("active_organization_id") or "").strip()
    if active:
        return active
    orgs = meta.get("organizations")
    if isinstance(orgs, dict) and len(orgs) == 1:
        return next(iter(orgs.keys()))
    return None


def list_linear_organizations() -> list[dict[str, Any]]:
    """Return connected Linear workspaces (organizations) with metadata only."""
    meta = _read_linear_metadata()
    orgs = meta.get("organizations")
    if not isinstance(orgs, dict):
        return []
    active = str(meta.get("active_organization_id") or "").strip()
    rows: list[dict[str, Any]] = []
    for org_id, entry in orgs.items():
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        row["id"] = str(org_id)
        row["active"] = str(org_id) == active
        url_key = str(row.get("url_key") or row.get("urlKey") or "").strip()
        row["key"] = url_key or str(row.get("name") or org_id)
        rows.append(row)
    rows.sort(key=lambda item: str(item.get("name") or item.get("key") or ""))
    return rows


def set_active_linear_organization(org_id: str) -> None:
    meta = _read_linear_metadata()
    orgs = meta.get("organizations")
    if not isinstance(orgs, dict) or org_id not in orgs:
        raise ProviderCredentialError(
            f"Linear workspace {org_id!r} is not connected. "
            "Run: potpie linear login --add"
        )
    org_entry = orgs[org_id]
    _write_metadata_entry(
        _LINEAR_CREDENTIALS_KEY,
        {
            **meta,
            "active_organization_id": org_id,
            "organization": {
                "id": org_id,
                "name": org_entry.get("name"),
                "url_key": org_entry.get("url_key") or org_entry.get("urlKey"),
            },
            "account": org_entry.get("account"),
            "expires_at": org_entry.get("expires_at"),
            "scopes": org_entry.get("scopes"),
        },
    )


def save_linear_organization_tokens(
    org_id: str,
    tokens: dict[str, Any],
    *,
    organization: dict[str, Any],
    account: dict[str, Any] | None = None,
) -> None:
    from context_engine.adapters.outbound.cli_auth.integration_profile import utc_now_iso

    prior = _read_linear_metadata()
    orgs = dict(prior.get("organizations") or {})
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    token_storage = "keychain"
    if access_token:
        token_storage = _store_keychain_secret(
            "Linear access token",
            _linear_access_token_secret(org_id),
            access_token,
        )
    if refresh_token:
        refresh_token_storage = _store_keychain_secret(
            "Linear refresh token",
            _linear_refresh_token_secret(org_id),
            refresh_token,
        )
        if refresh_token_storage == "file":
            token_storage = "file"

    scopes = tokens.get("scope") or tokens.get("scopes") or prior.get("scopes")
    org_entry: dict[str, Any] = {
        "id": org_id,
        "name": organization.get("name"),
        "url_key": organization.get("url_key") or organization.get("urlKey"),
        "account": account or organization.get("account"),
        "expires_at": tokens.get("expires_at"),
        "scopes": scopes,
        "stored_at": tokens.get("stored_at") or prior.get("stored_at"),
    }
    orgs[org_id] = org_entry
    now = utc_now_iso()
    record: dict[str, Any] = {
        "provider": "linear",
        "provider_host": "linear.app",
        "auth_type": "oauth",
        "token_storage": token_storage,
        "token_type": tokens.get("token_type") or prior.get("token_type") or "Bearer",
        "organizations": orgs,
        "active_organization_id": org_id,
        "organization": {
            "id": org_id,
            "name": org_entry.get("name"),
            "url_key": org_entry.get("url_key"),
        },
        "account": org_entry.get("account"),
        "expires_at": org_entry.get("expires_at"),
        "scopes": org_entry.get("scopes"),
        "stored_at": org_entry.get("stored_at"),
        "created_at": prior.get("created_at") or now,
        "updated_at": now,
        "metadata": {"auth_flow": "pkce"},
    }
    if isinstance(prior.get("workspaces"), dict):
        record["workspaces"] = prior["workspaces"]
    _write_metadata_entry(_LINEAR_CREDENTIALS_KEY, record)


def get_linear_tokens(organization_id: str | None = None) -> dict[str, Any]:
    """Return Linear OAuth tokens for ``organization_id`` or the active workspace."""
    meta = _read_linear_metadata()
    if not meta:
        return {}
    org_id = str(organization_id or get_active_linear_organization_id() or "").strip()
    if not org_id:
        return {}
    orgs = meta.get("organizations")
    if not isinstance(orgs, dict):
        return {}
    org_entry = orgs.get(org_id)
    if not isinstance(org_entry, dict):
        return {}
    access_token = _load_keychain_secret(
        "Linear access token",
        _linear_access_token_secret(org_id),
    )
    if not access_token:
        access_token = _load_keychain_secret(
            "Linear access token",
            _LINEAR_ACCESS_TOKEN_SECRET,
        )
    if not access_token:
        return {}
    refresh_token = _load_keychain_secret(
        "Linear refresh token",
        _linear_refresh_token_secret(org_id),
    )
    if not refresh_token:
        refresh_token = _load_keychain_secret(
            "Linear refresh token",
            _LINEAR_REFRESH_TOKEN_SECRET,
        )
    payload: dict[str, Any] = {
        **meta,
        **org_entry,
        "organization_id": org_id,
        "organization": {
            "id": org_id,
            "name": org_entry.get("name"),
            "url_key": org_entry.get("url_key") or org_entry.get("urlKey"),
        },
        "access_token": access_token,
    }
    if refresh_token:
        payload["refresh_token"] = refresh_token
    return payload


def save_integration_tokens(provider: str, tokens: dict[str, Any]) -> None:
    """Store Linear OAuth tokens in keychain and metadata on disk."""
    key = _norm_integration_key(provider)
    if key != _LINEAR_CREDENTIALS_KEY:
        raise ValueError(f"{provider!r} does not use OAuth token storage.")

    from context_engine.adapters.outbound.cli_auth.integration_profile import fetch_linear_viewer

    access_token = str(tokens.get("access_token") or "").strip()
    if not access_token:
        raise ProviderCredentialError("Linear access token is required.")

    profile = fetch_linear_viewer(access_token)
    organization = profile.get("organization")
    if not isinstance(organization, dict) or not organization.get("id"):
        raise ProviderCredentialError(
            "Linear login succeeded but workspace metadata was unavailable."
        )
    org_id = str(organization["id"])
    save_linear_organization_tokens(
        org_id,
        tokens,
        organization=organization,
        account=profile.get("account"),
    )


def write_integration_tokens(provider: str, tokens: dict[str, Any]) -> None:
    save_integration_tokens(provider, tokens)


def _norm_atlassian_product(product: str) -> str:
    key = str(product or "").strip().lower()
    if key in {"wiki", "conf"}:
        return "confluence"
    if key not in {"jira", "confluence"}:
        raise ValueError(
            f"Unknown Atlassian product {product!r} (expected 'jira' or 'confluence')."
        )
    return key


def _legacy_atlassian_metadata() -> dict[str, Any]:
    return _read_metadata_entry(_ATLASSIAN_CREDENTIALS_KEY)


def _product_metadata_key(product: str) -> str:
    key = _norm_atlassian_product(product)
    return _JIRA_CREDENTIALS_KEY if key == "jira" else _CONFLUENCE_CREDENTIALS_KEY


def _product_secret_name(product: str) -> tuple[str, str]:
    key = _norm_atlassian_product(product)
    if key == "jira":
        return _JIRA_TOKEN_SECRET, "Jira API token"
    return _CONFLUENCE_TOKEN_SECRET, "Confluence API token"


def save_jira_credentials(credentials: dict[str, Any]) -> None:
    _save_atlassian_product_credentials("jira", credentials)


def get_jira_credentials() -> dict[str, Any]:
    return _get_atlassian_product_credentials("jira")


def clear_jira_credentials() -> None:
    _clear_atlassian_product_credentials("jira")


def save_confluence_credentials(credentials: dict[str, Any]) -> None:
    _save_atlassian_product_credentials("confluence", credentials)


def get_confluence_credentials() -> dict[str, Any]:
    return _get_atlassian_product_credentials("confluence")


def clear_confluence_credentials() -> None:
    _clear_atlassian_product_credentials("confluence")


def _save_atlassian_product_credentials(
    product: str, credentials: dict[str, Any]
) -> None:
    from context_engine.adapters.outbound.cli_auth.integration_profile import (
        atlassian_site_from_entry,
        build_product_integration_record,
    )

    key = _norm_atlassian_product(product)
    secret_name, label = _product_secret_name(key)
    api_token = str(credentials.get("api_token") or "").strip()
    if not api_token:
        raise ProviderCredentialError(f"{key.capitalize()} API token is required.")

    token_storage = _store_keychain_secret(label, secret_name, api_token)
    prior = _read_metadata_entry(_product_metadata_key(key))
    merged = {**prior, **credentials}
    site = atlassian_site_from_entry(merged)
    if site.get("site_url") and not merged.get("site_url"):
        merged["site_url"] = site["site_url"]
    record = build_product_integration_record(key, merged)
    record["token_storage"] = token_storage
    _write_metadata_entry(_product_metadata_key(key), record)


def _get_atlassian_product_credentials(product: str) -> dict[str, Any]:
    key = _norm_atlassian_product(product)
    metadata = _read_metadata_entry(_product_metadata_key(key))
    if not metadata:
        legacy = _legacy_atlassian_metadata()
        if legacy:
            metadata = dict(legacy)
    if not metadata:
        return {}

    secret_name, label = _product_secret_name(key)
    api_token = _load_keychain_secret(label, secret_name)
    if not api_token:
        api_token = _load_keychain_secret(
            "Atlassian API token",
            _ATLASSIAN_LEGACY_TOKEN_SECRET,
        )
    if not api_token:
        return {}
    return {**metadata, "api_token": api_token}


def _clear_shared_atlassian_legacy_credentials() -> None:
    """Remove legacy shared Atlassian keychain secret and metadata (both products)."""
    _delete_keychain_secret("Atlassian API token", _ATLASSIAN_LEGACY_TOKEN_SECRET)
    _clear_metadata_entries(_ATLASSIAN_CREDENTIALS_KEY)


def _clear_atlassian_product_credentials(product: str) -> None:
    key = _norm_atlassian_product(product)
    secret_name, label = _product_secret_name(key)
    _delete_keychain_secret(label, secret_name)
    _clear_metadata_entries(_product_metadata_key(key))


def save_atlassian_credentials(credentials: dict[str, Any]) -> None:
    """Legacy combined Atlassian record retained for compatibility."""
    from context_engine.adapters.outbound.cli_auth.integration_profile import (
        atlassian_site_from_entry,
        build_atlassian_integration_record,
    )

    api_token = str(credentials.get("api_token") or "").strip()
    if not api_token:
        raise ProviderCredentialError("Atlassian API token is required.")
    token_storage = _store_keychain_secret(
        "Atlassian API token",
        _ATLASSIAN_LEGACY_TOKEN_SECRET,
        api_token,
    )
    prior = _legacy_atlassian_metadata()
    merged = {**prior, **credentials}
    site = atlassian_site_from_entry(merged)
    if site.get("site_url") and not merged.get("site_url"):
        merged["site_url"] = site["site_url"]
    record = build_atlassian_integration_record(merged)
    record["token_storage"] = token_storage
    _write_metadata_entry(_ATLASSIAN_CREDENTIALS_KEY, record)


def get_atlassian_credentials() -> dict[str, Any]:
    jira = get_jira_credentials()
    if jira:
        return jira
    confluence = get_confluence_credentials()
    if confluence:
        return confluence

    metadata = _legacy_atlassian_metadata()
    if not metadata:
        return {}
    api_token = _load_keychain_secret(
        "Atlassian API token",
        _ATLASSIAN_LEGACY_TOKEN_SECRET,
    )
    if not api_token:
        return {}
    return {**metadata, "api_token": api_token}


def clear_atlassian_credentials() -> None:
    _clear_shared_atlassian_legacy_credentials()
    _delete_keychain_secret("Jira API token", _JIRA_TOKEN_SECRET)
    _delete_keychain_secret("Confluence API token", _CONFLUENCE_TOKEN_SECRET)
    _clear_metadata_entries(
        _JIRA_CREDENTIALS_KEY,
        _CONFLUENCE_CREDENTIALS_KEY,
    )


def save_jira_workspace_prefs(*, project_key: str) -> None:
    prior = get_jira_credentials()
    if not prior.get("api_token"):
        raise ProviderCredentialError(
            "Jira is not connected. Run: potpie jira login"
        )
    workspaces = dict(prior.get("workspaces") or {})
    workspaces["jira_project"] = project_key.strip().upper()
    save_jira_credentials({**prior, "workspaces": workspaces})


def save_confluence_workspace_prefs(*, space_key: str) -> None:
    prior = get_confluence_credentials()
    if not prior.get("api_token"):
        raise ProviderCredentialError(
            "Confluence is not connected. Run: potpie confluence login"
        )
    workspaces = dict(prior.get("workspaces") or {})
    workspaces["confluence_space"] = space_key.strip().upper()
    save_confluence_credentials({**prior, "workspaces": workspaces})


def save_linear_workspace_prefs(
    *,
    organization_id: str,
    organization_key: str | None = None,
    team_key: str,
    team_id: str | None = None,
) -> None:
    metadata = _read_linear_metadata()
    if not metadata:
        raise ProviderCredentialError(
            "Linear is not connected. Run: potpie linear login"
        )
    if not get_linear_tokens(organization_id).get("access_token"):
        raise ProviderCredentialError(
            f"Linear token not found in {_integration_secret_store_label()}. "
            "Run: potpie linear login"
        )
    workspaces = dict(metadata.get("workspaces") or {})
    workspaces["linear_organization_id"] = organization_id.strip()
    if organization_key:
        workspaces["linear_organization_key"] = organization_key.strip()
    workspaces["linear_team"] = team_key.strip().upper()
    if team_id:
        workspaces["linear_team_id"] = team_id.strip()
    _write_metadata_entry(_LINEAR_CREDENTIALS_KEY, {**metadata, "workspaces": workspaces})


def save_atlassian_workspace_prefs(
    *,
    jira_project: str | None = None,
    confluence_space: str | None = None,
) -> None:
    if jira_project:
        save_jira_workspace_prefs(project_key=jira_project)
    if confluence_space:
        save_confluence_workspace_prefs(space_key=confluence_space)


def get_integration_tokens(provider: str) -> dict[str, Any]:
    """Return integration credentials with secrets loaded from keychain."""
    key = _norm_integration_key(provider)
    if key == _LINEAR_CREDENTIALS_KEY:
        return get_linear_tokens()
    if key in {
        _ATLASSIAN_CREDENTIALS_KEY,
        _JIRA_CREDENTIALS_KEY,
        _CONFLUENCE_CREDENTIALS_KEY,
    }:
        if key == _ATLASSIAN_CREDENTIALS_KEY:
            creds = get_atlassian_credentials()
        elif key == _JIRA_CREDENTIALS_KEY:
            creds = get_jira_credentials()
        else:
            creds = get_confluence_credentials()
        return {"auth_type": "api_token", **creds} if creds else {}

    raise ValueError(f"Unknown integration provider {provider!r}.")


def clear_integration_tokens(provider: str) -> None:
    """Remove stored credentials for an integration provider."""
    key = _norm_integration_key(provider)
    if key == _LINEAR_CREDENTIALS_KEY:
        meta = _read_linear_metadata()
        orgs = meta.get("organizations")
        if isinstance(orgs, dict):
            for org_id in orgs:
                _delete_keychain_secret(
                    "Linear access token",
                    _linear_access_token_secret(org_id),
                )
                _delete_keychain_secret(
                    "Linear refresh token",
                    _linear_refresh_token_secret(org_id),
                )
        _delete_keychain_secret("Linear access token", _LINEAR_ACCESS_TOKEN_SECRET)
        _delete_keychain_secret("Linear refresh token", _LINEAR_REFRESH_TOKEN_SECRET)
        _clear_metadata_entries(_LINEAR_CREDENTIALS_KEY)
        return
    if key == _JIRA_CREDENTIALS_KEY:
        clear_jira_credentials()
        return
    if key == _CONFLUENCE_CREDENTIALS_KEY:
        clear_confluence_credentials()
        return
    if key == _ATLASSIAN_CREDENTIALS_KEY:
        clear_atlassian_credentials()
        return
    raise ValueError(f"Unknown integration provider {provider!r}.")


def list_integration_providers() -> list[str]:
    integrations = list_integration_metadata()
    found: list[str] = []
    for key in (
        _LINEAR_CREDENTIALS_KEY,
        _JIRA_CREDENTIALS_KEY,
        _CONFLUENCE_CREDENTIALS_KEY,
        _ATLASSIAN_CREDENTIALS_KEY,
    ):
        if isinstance(integrations.get(key), dict):
            found.append(key)
    return found


def get_integration_status(provider: str) -> dict[str, Any]:
    from context_engine.adapters.outbound.cli_auth.integration_profile import (
        atlassian_account_from_entry,
        atlassian_site_from_entry,
        linear_account_from_entry,
    )

    key = _norm_integration_key(provider)

    if key == _LINEAR_CREDENTIALS_KEY:
        meta = _read_linear_metadata()
        orgs = meta.get("organizations")
        if not isinstance(orgs, dict) or not orgs:
            legacy = _load_keychain_secret(
                "Linear access token",
                _LINEAR_ACCESS_TOKEN_SECRET,
            )
            if not legacy:
                return {"provider": key, "authenticated": False, "auth_type": "oauth"}
        elif not any(
            get_linear_tokens(org_id).get("access_token") for org_id in orgs.keys()
        ):
            return {"provider": key, "authenticated": False, "auth_type": "oauth"}

        tokens = get_linear_tokens()
        if not tokens.get("access_token"):
            return {"provider": key, "authenticated": False, "auth_type": "oauth"}
        entry = _read_linear_metadata()
        account = linear_account_from_entry(entry)
        organization = entry.get("organization")
        org_name = organization.get("name") if isinstance(organization, dict) else None
        scopes = entry.get("scopes")
        scope = entry.get("scope")
        if scopes is None and scope is not None:
            scopes = scope
        connected = list_linear_organizations()
        return {
            "provider": key,
            "authenticated": True,
            "auth_type": "oauth",
            "login": account.get("name") or account.get("email"),
            "email": account.get("email"),
            "site_name": org_name,
            "expires_at": entry.get("expires_at"),
            "scope": scopes,
            "cloud_id": entry.get("cloud_id"),
            "stored_at": entry.get("stored_at"),
            "token_storage": entry.get("token_storage"),
            "workspace_count": len(connected),
        }

    if key == "github":
        entry = _read_metadata_entry("github")
        if not entry:
            return {"provider": key, "authenticated": False, "auth_type": "oauth"}
        token = _load_keychain_secret("GitHub token", _GITHUB_TOKEN_SECRET)
        if not token:
            return {"provider": key, "authenticated": False, "auth_type": "oauth"}
        account = entry.get("account")
        account_dict = dict(account) if isinstance(account, dict) else {}
        scopes = entry.get("scopes")
        return {
            "provider": key,
            "authenticated": True,
            "auth_type": "oauth",
            "login": account_dict.get("login"),
            "email": account_dict.get("email"),
            "scope": scopes,
            "stored_at": entry.get("updated_at") or entry.get("created_at"),
            "token_storage": entry.get("token_storage"),
            "provider_host": entry.get("provider_host"),
        }

    if key in {
        _ATLASSIAN_CREDENTIALS_KEY,
        _JIRA_CREDENTIALS_KEY,
        _CONFLUENCE_CREDENTIALS_KEY,
    }:
        if key == _ATLASSIAN_CREDENTIALS_KEY:
            creds = get_atlassian_credentials()
            entry = _legacy_atlassian_metadata() or creds
        elif key == _JIRA_CREDENTIALS_KEY:
            creds = get_jira_credentials()
            entry = (
                _read_metadata_entry(_JIRA_CREDENTIALS_KEY)
                or _legacy_atlassian_metadata()
            )
        else:
            creds = get_confluence_credentials()
            entry = (
                _read_metadata_entry(_CONFLUENCE_CREDENTIALS_KEY)
                or _legacy_atlassian_metadata()
            )

        site = atlassian_site_from_entry(entry or creds)
        if not creds or not site.get("site_url"):
            return {"provider": key, "authenticated": False, "auth_type": "api_token"}

        account = atlassian_account_from_entry(entry or creds)
        return {
            "provider": key,
            "authenticated": True,
            "auth_type": "api_token",
            "email": account.get("email") or (entry or creds).get("email"),
            "site_url": site.get("site_url") or (entry or creds).get("site_url"),
            "site_name": site.get("site_name") or (entry or creds).get("site_name"),
            "cloud_id": site.get("cloud_id") or (entry or creds).get("cloud_id"),
            "stored_at": (entry or creds).get("stored_at"),
            "token_storage": (entry or creds).get("token_storage"),
        }

    raise ValueError(f"Unknown integration provider {provider!r}.")


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
    previous_metadata = get_integration_metadata(key)
    previous_token = (
        _load_keychain_secret("GitHub token", _GITHUB_TOKEN_SECRET)
        if previous_metadata
        else ""
    )
    stored_payload["token_storage"] = _store_keychain_secret(
        "GitHub token",
        _GITHUB_TOKEN_SECRET,
        access_token,
    )
    try:
        write_integration_metadata(key, stored_payload)
    except Exception:
        try:
            if previous_metadata and previous_token:
                _store_keychain_secret(
                    "GitHub token",
                    _GITHUB_TOKEN_SECRET,
                    previous_token,
                )
            else:
                _delete_keychain_secret("GitHub token", _GITHUB_TOKEN_SECRET)
        except Exception:
            pass
        raise


def get_provider_credentials(provider: str) -> dict[str, Any]:
    """Return provider metadata merged with secrets from keychain."""
    key = _norm_integration_key(provider)
    if key != "github":
        return {}
    metadata = get_integration_metadata(key)
    if not metadata:
        return {}
    result = dict(metadata)
    token = _load_keychain_secret("GitHub token", _GITHUB_TOKEN_SECRET)
    if not token:
        token_storage = str(result.get("token_storage") or "").strip()
        raise ProviderCredentialError(
            f"GitHub token not found in {_storage_label(token_storage)}. "
            "Run: potpie github login"
        )
    result["access_token"] = token
    return result


def clear_provider_credentials(provider: str) -> None:
    """Remove provider secrets from keychain and drop integration metadata."""
    key = _norm_integration_key(provider)
    if key != "github":
        raise ValueError(f"Unsupported provider {provider!r}; expected 'github'.")
    _delete_keychain_secret("GitHub token", _GITHUB_TOKEN_SECRET)
    clear_integration_metadata(key)
