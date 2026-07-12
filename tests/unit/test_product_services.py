from __future__ import annotations

from pathlib import Path

from potpie.auth import AccountAuthService, IntegrationAuthService
from potpie.config import ProductConfigService
from potpie.runtime.composition import engine_actor_for_identity
from potpie.runtime.settings import ProductSettings
from tests._auth_fakes import InMemoryCredentialStore


def test_account_auth_service_owns_identity_and_credentials() -> None:
    credentials = InMemoryCredentialStore()
    service = AccountAuthService(credentials)

    anonymous = service.whoami()
    authenticated = service.login_api_key(
        "secret-api-key", api_url="https://potpie.test"
    )
    actor = engine_actor_for_identity(authenticated)
    service.logout()

    assert anonymous.authenticated is False
    assert authenticated.authenticated is True
    assert authenticated.auth_type == "api_key"
    assert credentials.api_base_url == "https://potpie.test"
    assert actor.subject == "potpie-account"
    assert actor.auth_mode == "api_key"
    assert service.whoami().authenticated is False


def test_integration_auth_service_redacts_tokens_and_clears_provider() -> None:
    credentials = InMemoryCredentialStore()
    credentials.write_provider_credentials(
        "github",
        {
            "access_token": "github-secret",
            "account": {"login": "octocat"},
        },
    )
    service = IntegrationAuthService(credentials)

    connected = service.status("github")
    service.logout("github")

    assert connected.connected is True
    assert "access_token" not in connected.details
    assert service.status("github").connected is False


def test_product_config_is_atomic_public_and_loaded_by_product_settings(
    tmp_path: Path,
) -> None:
    service = ProductConfigService(tmp_path)

    service.set("backend", "neo4j")
    service.set("runtime_mode", "in-process")
    service.set("provider.api_key", "secret")
    loaded = ProductSettings.load(environ={"POTPIE_HOME": str(tmp_path)})

    assert service.get("backend") == "neo4j"
    assert loaded.backend == "neo4j"
    assert loaded.runtime_mode == "in-process"
    assert service.list_public()["provider.api_key"] == "<redacted>"
    assert not service.path.with_suffix(".tmp").exists()


def test_context_engine_has_no_root_auth_imports() -> None:
    root = Path(__file__).parents[2] / "potpie" / "context-engine" / "src"
    offenders = [
        path
        for path in root.rglob("*.py")
        if "from potpie.auth" in path.read_text(encoding="utf-8")
        or "import potpie.auth" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
