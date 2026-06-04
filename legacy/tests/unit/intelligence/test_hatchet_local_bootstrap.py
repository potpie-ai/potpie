"""Unit tests for the local Hatchet bootstrap helper.

Shell-command boundaries (docker/psql) are mocked via an injected runner so these
run without Docker, Postgres, or a live Hatchet server.
"""

import pytest

from app.modules.intelligence.agents import hatchet_local_bootstrap as hb


class FakeResult:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class FakeRunner:
    """Records argv lists and returns canned results by substring match."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = responses or []  # list of (substr, FakeResult)

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        joined = " ".join(cmd)
        for substr, result in self._responses:
            if substr in joined:
                return result
        return FakeResult()

    def commands(self):
        return [" ".join(c) for c in self.calls]


# ── write_env_file ────────────────────────────────────────────────────────


def test_write_env_file_contains_token_and_client_vars(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    hb.write_env_file(str(env_path), "tok-abc123")
    content = env_path.read_text()
    assert "HATCHET_CLIENT_TOKEN=tok-abc123" in content
    assert "HATCHET_CLIENT_HOST_PORT=localhost:7077" in content
    assert "HATCHET_CLIENT_SERVER_URL=http://localhost:8080" in content
    assert "HATCHET_CLIENT_TLS_STRATEGY=none" in content


# ── token_present_in_env ──────────────────────────────────────────────────


def test_token_present_in_env_true_when_token_has_value(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    env_path.write_text(
        "HATCHET_CLIENT_TOKEN=abc\nHATCHET_CLIENT_HOST_PORT=localhost:7077\n"
    )
    assert hb.token_present_in_env(str(env_path)) is True


def test_token_present_in_env_false_when_file_missing(tmp_path):
    assert hb.token_present_in_env(str(tmp_path / "nope.env")) is False


def test_token_present_in_env_false_when_token_empty(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    env_path.write_text("HATCHET_CLIENT_TOKEN=\n")
    assert hb.token_present_in_env(str(env_path)) is False


# ── ensure_hatchet_database (idempotent) ──────────────────────────────────


def test_ensure_database_idempotent_skips_create_when_exists():
    runner = FakeRunner(
        responses=[("SELECT 1 FROM pg_database", FakeResult(stdout="1\n"))]
    )
    created = hb.ensure_hatchet_database(run=runner)
    assert created is False
    assert not any("CREATE DATABASE" in c for c in runner.commands())


def test_ensure_database_creates_when_absent():
    runner = FakeRunner(
        responses=[("SELECT 1 FROM pg_database", FakeResult(stdout="\n"))]
    )
    created = hb.ensure_hatchet_database(run=runner)
    assert created is True
    assert any("CREATE DATABASE" in c and "hatchet" in c for c in runner.commands())


def test_ensure_database_raises_when_select_fails():
    runner = FakeRunner(
        responses=[
            ("SELECT 1 FROM pg_database", FakeResult(returncode=1, stdout="")),
        ]
    )
    with pytest.raises(RuntimeError, match="Failed to query database list"):
        hb.ensure_hatchet_database(run=runner)


def test_ensure_database_raises_when_create_fails():
    runner = FakeRunner(
        responses=[
            ("SELECT 1 FROM pg_database", FakeResult(stdout="\n")),  # DB absent
            ("CREATE DATABASE", FakeResult(returncode=1, stdout="")),
        ]
    )
    with pytest.raises(RuntimeError, match="Failed to create database"):
        hb.ensure_hatchet_database(run=runner)


def test_wait_for_postgres_raises_when_never_ready():
    runner = FakeRunner(
        responses=[("pg_isready", FakeResult(returncode=1))]
    )
    with pytest.raises(RuntimeError, match="not ready after"):
        hb._wait_for_postgres(run=runner, retries=2, delay=0)


def test_bootstrap_raises_when_postgres_compose_fails(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    runner = FakeRunner(
        responses=[
            ("compose up -d postgres", FakeResult(returncode=1, stdout="")),
        ]
    )
    with pytest.raises(RuntimeError, match="Failed to start postgres"):
        hb.bootstrap(run=runner, env_path=str(env_path), do_compose=True)


def test_bootstrap_raises_when_hatchet_compose_fails(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    runner = FakeRunner(
        responses=[
            ("compose up -d postgres", FakeResult(returncode=0)),
            ("pg_isready", FakeResult(returncode=0)),
            ("SELECT 1 FROM pg_database", FakeResult(stdout="1\n")),
            ("--profile hatchet up", FakeResult(returncode=1, stdout="")),
        ]
    )
    with pytest.raises(RuntimeError, match="Failed to start hatchet services"):
        hb.bootstrap(run=runner, env_path=str(env_path), do_compose=True)


# ── create_api_token ──────────────────────────────────────────────────────


def test_create_api_token_extracts_token_and_uses_tenant():
    runner = FakeRunner(
        responses=[("token create", FakeResult(stdout="\neyJhbGci.JWT.token\n"))]
    )
    token = hb.create_api_token(run=runner)
    assert token == "eyJhbGci.JWT.token"
    assert any("token create" in c and "707d0855" in c for c in runner.commands())


# ── bootstrap orchestration ───────────────────────────────────────────────


def test_bootstrap_skips_token_when_already_present(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    env_path.write_text("HATCHET_CLIENT_TOKEN=existing-token\n")
    runner = FakeRunner(
        responses=[("SELECT 1 FROM pg_database", FakeResult(stdout="1\n"))]
    )
    hb.bootstrap(run=runner, env_path=str(env_path), do_compose=False)
    assert not any("token create" in c for c in runner.commands())
    assert "existing-token" in env_path.read_text()


def test_bootstrap_creates_token_and_writes_env_when_absent(tmp_path):
    env_path = tmp_path / ".env.hatchet.local"
    runner = FakeRunner(
        responses=[
            ("SELECT 1 FROM pg_database", FakeResult(stdout="\n")),  # DB absent
            ("token create", FakeResult(stdout="my-jwt-token\n")),
        ]
    )
    hb.bootstrap(run=runner, env_path=str(env_path), do_compose=False)
    assert any("CREATE DATABASE" in c for c in runner.commands())
    assert any("token create" in c for c in runner.commands())
    assert "HATCHET_CLIENT_TOKEN=my-jwt-token" in env_path.read_text()
