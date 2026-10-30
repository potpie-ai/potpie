"""Current provider commands, status reporting, and read-only verification."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from typer.testing import CliRunner

from potpie.auth.adapters import integration_verify as verify
from potpie.auth.adapters import provider_config as provider
from potpie.auth.adapters.credentials_store import ProviderCredentialError
from potpie.auth.adapters.http import AuthHttpError
from potpie.auth.adapters.integration_profile import fetch_linear_viewer
from potpie.cli import main as cli
from potpie.cli.auth import auth_commands as commands
from potpie.cli.commands import _common
from tests._auth_fakes import FakeAuthHttpClient, InMemoryCredentialStore

RUNNER = CliRunner()
PROVIDERS = ("github", "linear", "jira", "confluence", "gitlab", "gitbucket")


def _status(name, authenticated):
    return {
        "provider": name,
        "authenticated": authenticated,
        "auth_type": "oauth" if name in {"github", "linear"} else "api_token",
        "login": "Ada",
        "email": "ada@example.com",
        "site_name": "Team",
        "site_url": "https://team.atlassian.net",
        "expires_at": 9999999999.0,
        "cloud_id": "cloud-1",
        "token_storage": "file",
    }


@pytest.fixture
def provider_cli(monkeypatch):
    monkeypatch.setattr(commands, "ensure_runtime_environment_loaded", lambda: None)
    store = InMemoryCredentialStore()
    _common.set_store(store)
    return store


@pytest.mark.parametrize("selected", ["linear", "jira", "github"])
def test_status_reports_every_provider_with_one_authenticated_row(
    monkeypatch, provider_cli, selected
):
    monkeypatch.setattr(
        commands, "get_integration_status", lambda name: _status(name, name == selected)
    )
    result = RUNNER.invoke(cli.app, ["--json", "auth", "status"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)["integrations"]
    assert [row["provider"] for row in rows] == list(PROVIDERS)
    assert [row["provider"] for row in rows if row["authenticated"]] == [selected]
    assert next(row for row in rows if row["provider"] == selected) == _status(
        selected, True
    )


@pytest.mark.parametrize("selected", ["linear", "jira", "github"])
@pytest.mark.parametrize("outcome", ["ok", "refused", "credential-error"])
def test_status_verification_uses_the_provider_credential_port(
    monkeypatch, provider_cli, selected, outcome
):
    monkeypatch.setattr(
        commands, "get_integration_status", lambda name: _status(name, name == selected)
    )
    ports, probes = [], []

    def credentials(name):
        ports.append(name)
        if outcome == "credential-error":
            raise ProviderCredentialError("credential file unavailable")
        return {"access_token": "token", "api_token": "token"}

    def probe(name, payload):
        probes.append((name, payload))
        return outcome == "ok", "ok (Ada)" if outcome == "ok" else "gateway unavailable"

    monkeypatch.setattr(commands, "ensure_valid_integration_tokens", credentials)
    monkeypatch.setattr(commands, "get_integration_tokens", credentials)
    provider_cli.get_provider_credentials = credentials
    monkeypatch.setattr(commands, "verify_integration_access", probe)
    result = RUNNER.invoke(cli.app, ["--json", "auth", "status", "--verify"])
    assert result.exit_code == 0, result.output
    row = next(
        row
        for row in json.loads(result.stdout)["integrations"]
        if row["provider"] == selected
    )
    assert ports == [selected]
    assert row["verified"] is (outcome == "ok")
    expected = {
        "ok": "ok (Ada)",
        "refused": "gateway unavailable",
        "credential-error": "credential file unavailable",
    }[outcome]
    assert row["verify_message"] == expected
    assert len(probes) == (0 if outcome == "credential-error" else 1)


@pytest.mark.parametrize("as_json", [False, True], ids=["human", "json"])
def test_one_provider_metadata_error_does_not_hide_other_status_rows(
    monkeypatch, provider_cli, as_json
):
    def status(name):
        if name == "linear":
            raise ProviderCredentialError("credential file unavailable")
        return _status(name, False)

    monkeypatch.setattr(commands, "get_integration_status", status)
    result = RUNNER.invoke(
        cli.app, (["--json"] if as_json else []) + ["auth", "status"]
    )
    assert result.exit_code == 0, result.output
    if as_json:
        rows = json.loads(result.stdout)["integrations"]
        assert len(rows) == len(PROVIDERS)
        assert next(row for row in rows if row["provider"] == "linear") == {
            "provider": "linear",
            "authenticated": False,
            "status_error": "credential file unavailable",
        }
    else:
        assert "linear: status unavailable" in result.stdout
        assert "github: not authenticated" in result.stdout


@pytest.mark.parametrize(
    "refresh_error", [False, True], ids=["refreshed-expiry", "metadata-read-error"]
)
def test_linear_status_refreshes_metadata_without_losing_the_row(
    monkeypatch, provider_cli, refresh_error
):
    reads = []

    def status(name):
        if name != "linear":
            return _status(name, False)
        reads.append(name)
        if len(reads) == 2 and refresh_error:
            raise ProviderCredentialError("credential file unavailable")
        return {**_status(name, True), "expires_at": len(reads)}

    monkeypatch.setattr(commands, "get_integration_status", status)
    monkeypatch.setattr(
        commands,
        "ensure_valid_integration_tokens",
        lambda _name: {"access_token": "token"},
    )
    monkeypatch.setattr(
        commands, "verify_integration_access", lambda *_args: (True, "ok (Ada)")
    )
    result = RUNNER.invoke(cli.app, ["--json", "auth", "status", "--verify"])
    assert result.exit_code == 0, result.output
    row = next(
        row
        for row in json.loads(result.stdout)["integrations"]
        if row["provider"] == "linear"
    )
    assert reads == ["linear", "linear"]
    assert row["verified"] is (not refresh_error)
    assert row["verify_message"] == (
        "credential file unavailable" if refresh_error else "ok (Ada)"
    )
    assert row["expires_at"] == (1 if refresh_error else 2)


@pytest.mark.parametrize("selected", ["linear", "jira"])
@pytest.mark.parametrize("success", [False, True], ids=["refused", "verified"])
def test_human_status_preserves_identity_expiry_and_verification_details(
    monkeypatch, provider_cli, selected, success
):
    monkeypatch.setattr(
        commands, "get_integration_status", lambda name: _status(name, name == selected)
    )
    monkeypatch.setattr(
        commands,
        "ensure_valid_integration_tokens",
        lambda *_args: {"access_token": "token"},
    )
    monkeypatch.setattr(
        commands, "get_integration_tokens", lambda *_args: {"api_token": "token"}
    )
    monkeypatch.setattr(
        commands,
        "verify_integration_access",
        lambda *_args: (success, "ok (Ada)" if success else "gateway unavailable"),
    )
    result = RUNNER.invoke(cli.app, ["auth", "status", "--verify"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.stdout.split())
    for fragment in (
        f"{selected}: authenticated",
        "login=Ada",
        "email=ada@example.com",
        "site=Team",
        "expires_at=",
        "cloud_id=cloud-1",
        "token_storage=file",
    ):
        assert fragment in text
    assert (
        "verify=ok (Ada)" if success else "verify failed (gateway unavailable)"
    ) in text


@pytest.mark.parametrize(
    ("selected", "authenticated", "error", "exit_code"),
    [
        ("linear", True, False, 0),
        ("linear", False, False, 1),
        ("linear", True, True, 4),
        ("jira", True, False, 0),
        ("jira", False, False, 0),
        ("jira", True, True, 4),
        ("confluence", True, False, 0),
    ],
    ids=[
        "linear",
        "linear-disconnected",
        "linear-storage-error",
        "jira",
        "jira-stale",
        "jira-storage-error",
        "confluence",
    ],
)
def test_provider_logout_dispatch_and_storage_failures(
    monkeypatch, provider_cli, selected, authenticated, error, exit_code
):
    monkeypatch.setattr(
        commands, "get_integration_status", lambda name: _status(name, authenticated)
    )
    cleared = []

    def clear(name):
        cleared.append(name)
        if error:
            raise ProviderCredentialError("credential file unavailable")

    provider_cli.clear_integration_tokens = clear
    result = RUNNER.invoke(cli.app, ["--json", selected, "logout"])
    assert result.exit_code == exit_code, result.output
    assert cleared == ([] if selected == "linear" and not authenticated else [selected])
    if exit_code == 0:
        payload = json.loads(result.stdout)
        assert payload["ok"] is True and payload["provider"] == selected
        if not authenticated:
            assert payload["cleared_stale"] is True


@pytest.mark.parametrize("method", ["ls", "select"])
def test_linear_read_commands_preserve_workspace_and_issue_output(
    monkeypatch, provider_cli, method
):
    if method == "ls":
        monkeypatch.setattr(
            commands,
            "fetch_linear_workspaces",
            lambda limit=10: [
                {"id": "org-1", "name": "Potpie AI CLI", "key": "potpie-ai-cli"}
            ],
        )
        args = ["linear", "ls"]
    else:
        monkeypatch.setattr(
            commands,
            "run_linear_use_flow",
            lambda **_kwargs: {
                "product": "linear",
                "workspace_key": "potpie-ai-cli",
                "workspace_name": "Potpie AI CLI",
                "team_key": "ENG",
                "team_name": "Engineering",
                "items": [
                    {
                        "identifier": "ENG-1",
                        "title": "Fix login",
                        "status": "In Progress",
                        "url": "https://linear.app/issue/ENG-1",
                    }
                ],
            },
        )
        args = ["linear", "select", "--org", "potpie-ai-cli", "--key", "ENG"]
    result = RUNNER.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "Potpie AI CLI" in result.stdout
    if method == "select":
        assert "ENG-1" in result.stdout and "Fix login" in result.stdout


@pytest.mark.parametrize("product", ["linear", "jira", "confluence"])
def test_already_connected_responses_are_structured(monkeypatch, provider_cli, product):
    _common.set_json(True)
    printed = []
    monkeypatch.setattr(
        commands,
        "print_plain_line",
        lambda *args, **kwargs: printed.append(kwargs.get("json_payload")),
    )
    commands._handle_already_connected(product, _status(product, True))
    assert printed[-1]["already_connected"] is True
    assert printed[-1]["provider"] == product


@pytest.mark.parametrize("as_json", [False, True], ids=["human", "json"])
@pytest.mark.parametrize("product", ["jira", "confluence"])
def test_product_selection_results_preserve_workspace_and_item_details(
    monkeypatch, provider_cli, capsys, as_json, product
):
    _common.set_json(as_json)
    commands._run_product_use_result(
        {
            "product": product,
            "workspace_key": "DOCS",
            "workspace_name": "Documentation",
            "items": [
                {
                    "key": "DOCS-1",
                    "title": "Intro",
                    "summary": "Intro",
                    "space": "DOCS",
                    "url": "https://example.com/item",
                }
            ],
        },
        product_label=product.title(),
    )
    output = capsys.readouterr().out
    if as_json:
        payload = json.loads(output)
        assert payload["provider"] == product and payload["workspace_key"] == "DOCS"
        assert payload["items"][0]["title"] == "Intro"
    else:
        assert "Documentation" in output and "Intro" in output


def _viewer(with_org=True):
    result = {"id": "v1", "name": "Ada", "email": "ada@example.com"}
    if with_org:
        result["organization"] = {"id": "org1", "name": "Acme", "urlKey": "acme"}
    return {"data": {"viewer": result}}


@pytest.mark.parametrize(
    ("case", "status", "payload", "message", "success"),
    [
        ("valid", 200, _viewer(), "Ada @ Acme", True),
        ("without-organization", 200, _viewer(False), "Ada", True),
        ("http-error", 401, {}, "HTTP 401", False),
        (
            "graphql-error",
            200,
            {"errors": [{"message": "bad token"}]},
            "rejected",
            False,
        ),
        (
            "invalid-viewer",
            200,
            {"data": {"viewer": "invalid"}},
            "viewer unavailable",
            False,
        ),
        ("non-json", 200, None, "non-JSON", False),
        ("transport", 200, None, "request failed", False),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_linear_profile_and_verification_share_the_read_only_http_contract(
    case, status, payload, message, success
):
    def respond(method, url, **kwargs):
        assert method == "POST" and url == "https://api.linear.app/graphql"
        assert kwargs["headers"]["Authorization"] == "Bearer token"
        assert "viewer" in kwargs["json"]["query"]
        if case == "transport":
            raise AuthHttpError("offline")
        return (
            httpx.Response(status, text="not json")
            if case == "non-json"
            else httpx.Response(status, json=payload)
        )

    client = FakeAuthHttpClient(handler=respond)
    profile = fetch_linear_viewer("token", http=client)
    ok, detail = verify.verify_integration_access(
        "linear", {"access_token": "token"}, http=client
    )
    assert ok is success and message in detail
    if success:
        assert profile["account"] == {
            "id": "v1",
            "name": "Ada",
            "email": "ada@example.com",
        }
        if case == "valid":
            assert profile["organization"] == {
                "id": "org1",
                "name": "Acme",
                "url_key": "acme",
            }
    else:
        assert profile == {}
    assert len(client.calls) == 2 and client.closed is False


@pytest.mark.parametrize(
    ("credentials", "expected", "message", "probed"),
    [
        ({}, False, "not authenticated", False),
        ({"access_token": "token", "expires_at": 1}, False, "expired", False),
        ({"access_token": "token", "expires_at": "invalid"}, True, "ok (Ada)", True),
    ],
    ids=["missing-token", "expired", "unknown-expiry"],
)
def test_linear_verification_handles_missing_and_expired_tokens(
    monkeypatch, credentials, expected, message, probed
):
    calls = []
    monkeypatch.setattr(
        verify,
        "_verify_linear",
        lambda token, **kwargs: (calls.append(token) or True, "ok (Ada)"),
    )
    assert verify.verify_integration_access("linear", credentials) == (
        expected,
        "access token expired" if message == "expired" else message,
    )
    assert calls == (["token"] if probed else [])
    assert fetch_linear_viewer(" ") == {}


@pytest.mark.parametrize("product", ["jira", "confluence"])
def test_atlassian_verification_discovers_the_cloud_and_preserves_the_account(
    monkeypatch, product
):
    monkeypatch.setattr(verify, "fetch_cloud_id_for_site", lambda _url: "cloud-1")
    calls = []

    def gateway(email, token, cloud, selected, **_kwargs):
        calls.append((selected, cloud, email, token))
        return SimpleNamespace(ok=True, display_name="Ada", error_kind=None)

    monkeypatch.setattr(verify, "verify_gateway_product", gateway)
    ok, message = verify.verify_integration_access(
        product,
        {
            "email": "ada@example.com",
            "api_token": "token",
            "site_url": "https://team.atlassian.net",
        },
    )
    assert ok is True and "Ada" in message
    assert calls == [(product, "cloud-1", "ada@example.com", "token")]
    assert verify.verify_integration_access(product, {}) == (False, "not authenticated")


@pytest.mark.parametrize("provider_name", ["slack", "atlassian"])
def test_verification_requires_a_current_provider(provider_name):
    ok, message = verify.verify_integration_access(provider_name, {})
    assert ok is False and "unknown provider" in message


@pytest.mark.parametrize("invalid", [None, "not-a-number", 0.0, "future", "past"])
def test_expiry_classification_preserves_unknown_and_expired_states(invalid):
    value = (
        commands.time.time() + 3600
        if invalid == "future"
        else commands.time.time() - 10
        if invalid == "past"
        else invalid
    )
    assert commands._token_is_expired(value) is (invalid in {0.0, "past"})


@pytest.mark.parametrize(
    ("redirect", "override", "port", "candidates"),
    [
        (None, None, 28757, (28757, 28763)),
        ("http://127.0.0.1:9001/custom/callback", None, 9001, (9001,)),
        ("http://localhost:8080/callback", "9999", 9999, (9999,)),
        ("http://localhost:8080/callback", "not-a-port", 8080, (8080,)),
        ("http://localhost:8080/callback", "70000", 8080, (8080,)),
        ("http://localhost:8080/callback", "0", 8080, (8080,)),
    ],
    ids=["default", "redirect", "override", "invalid-override", "overflow", "zero"],
)
def test_oauth_callback_selection_is_local_and_has_explicit_fallbacks(
    monkeypatch, redirect, override, port, candidates
):
    for key, value in (
        ("POTPIE_CLI_OAUTH_REDIRECT_URI", redirect),
        ("POTPIE_CLI_OAUTH_CALLBACK_PORT", override),
    ):
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    assert provider.get_callback_port() == port
    assert provider.get_callback_port_candidates() == candidates
    assert provider.get_callback_host() in {"localhost", "127.0.0.1"}
    if redirect and not override:
        assert provider.get_redirect_uri() == redirect
        assert provider.get_callback_path() == "/custom/callback"
        assert (
            provider.get_redirect_uri_for_port(9002)
            == "http://127.0.0.1:9002/custom/callback"
        )


@pytest.mark.parametrize(
    "redirect", ["https://example.com/cb", "http://evil.example:8080/callback"]
)
def test_oauth_callbacks_refuse_remote_hosts(monkeypatch, redirect):
    monkeypatch.delenv("POTPIE_CLI_OAUTH_CALLBACK_PORT", raising=False)
    monkeypatch.setenv("POTPIE_CLI_OAUTH_REDIRECT_URI", redirect)
    with pytest.raises(ValueError, match="localhost"):
        provider.get_callback_host()


def test_linear_public_client_configuration_and_pkce_url(monkeypatch):
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    load_settings = provider.load_runtime_settings
    monkeypatch.setattr(
        provider,
        "load_runtime_settings",
        lambda: SimpleNamespace(linear_client_id="configured-id"),
    )
    assert provider.get_client_id("linear") == "configured-id"
    monkeypatch.setattr(provider, "load_runtime_settings", load_settings)
    monkeypatch.setenv("LINEAR_CLIENT_ID", "override-id")
    monkeypatch.delenv("LINEAR_OAUTH_SCOPE", raising=False)
    assert provider.get_client_id("linear") == "override-id"
    assert provider.get_scopes("linear") == "read"
    monkeypatch.setenv("LINEAR_OAUTH_SCOPE", "read,write")
    assert provider.get_scopes("linear") == "read,write"
    assert provider.get_client_secret("linear") == ""
    assert provider.token_url("linear") == provider.LINEAR_TOKEN_URL
    url = commands._build_linear_authorization_url(
        state="state-token",
        code_challenge="challenge",
        redirect_uri="http://localhost:9000/callback",
    )
    parsed = urlparse(url)
    assert (
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        == "https://linear.app/oauth/authorize"
    )
    query = parse_qs(parsed.query)
    assert query["client_id"] == ["override-id"]
    assert query["state"] == ["state-token"]
    assert query["code_challenge"] == ["challenge"]
    assert query["redirect_uri"] == ["http://localhost:9000/callback"]


@pytest.mark.parametrize(
    "lookup",
    [provider.authorization_url, provider.token_url],
    ids=["authorization", "token"],
)
def test_oauth_configuration_does_not_confuse_device_flow_with_pkce(lookup):
    with pytest.raises(ValueError, match="Unsupported OAuth provider"):
        lookup("github")


def test_linear_login_command_forwards_force_and_workspace_selection(
    monkeypatch, provider_cli
):
    calls = []
    monkeypatch.setattr(
        commands, "_run_linear_oauth_flow", lambda **kwargs: calls.append(kwargs)
    )
    result = RUNNER.invoke(cli.app, ["linear", "login", "--force", "--add"])
    assert result.exit_code == 0, result.output
    assert calls == [{"force": True, "add": True}]


@pytest.mark.parametrize("fails", [False, True], ids=["refreshed", "refresh-failure"])
def test_linear_session_refresh_handles_a_storage_failure(monkeypatch, fails):
    def tokens(_provider):
        if fails:
            raise ProviderCredentialError("credential file unavailable")
        return {"access_token": "refreshed-token"}

    monkeypatch.setattr(commands, "ensure_valid_integration_tokens", tokens)
    assert commands._try_refresh_linear_session() is (not fails)
