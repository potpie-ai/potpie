"""The CLI against a managed host over real HTTP, as the editor extension drives it.

The reference multi-actor host (``tests/support/reference_host.py``) listens on
loopback; the CLI reaches it through its registry exactly as against a deployed
host. The onboarding sequence runs the real entry point in a subprocess with
the key on stdin, and asserts the JSON fields the extension reads; engine
paths and authorization run in-process against the same host.

Alice administers ``pot_shop``, Bob reads it, Carol holds no grant.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.main import app
from potpie.config import host_registry
from potpie.runtime import remote
from tests.support.reference_host import ALICE_KEY, BOB_KEY, CAROL_KEY
from tests.support.reference_host_server import ServedReferenceHost

ROOT = Path(__file__).resolve().parents[2]
REMOTE_URL = "https://github.com/acme/shop.git"
REPO_KEY = "github.com/acme/shop"


@pytest.fixture(autouse=True)
def _scratch_user_home(tmp_path, monkeypatch) -> None:
    """The subprocess CLI gets a scratch ``HOME`` too, never the real one."""
    home = tmp_path / "user-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")


@pytest.fixture
def served(tmp_path):
    with ServedReferenceHost(tmp_path / "reference-host") as host:
        yield host
    remote._connections.clear()


@pytest.fixture
def clone(tmp_path, monkeypatch) -> Path:
    """A working tree whose ``origin`` remote names the shared repository."""
    repo = tmp_path / "shop"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", REMOTE_URL], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    return repo


def _cli(*args: str, cwd: Path, stdin: str | None = None) -> dict[str, Any]:
    """Run the installed entry point the way the extension does."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("POTPIE_MANAGED")
    }
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-m", "potpie.cli.main", *args],
        cwd=cwd,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout)


def test_the_extension_onboarding_sequence(served, clone) -> None:
    host_set = _cli(
        "--json",
        "host",
        "set",
        served.base_url,
        "--no-check",
        "--token",
        "-",
        cwd=clone,
        stdin=ALICE_KEY + "\n",
    )
    assert host_set["origin"] == "managed"
    assert host_set["endpoint"] == served.base_url
    assert host_set["checked"] is False
    assert host_set["shadowed_by_env"] is None

    assert _cli("--json", "host", "use", "managed", cwd=clone) == {
        "active_origin": "managed"
    }

    listed = _cli("--json", "pot", "list", "--managed", cwd=clone)
    assert [(row["id"], row["name"], row["archived"]) for row in listed["pots"]] == [
        ("pot_shop", "shop", False),
        ("pot_lab", "lab", False),
    ]
    assert {row["origin"] for row in listed["pots"]} == {"managed"}

    info = _cli("--json", "pot", "info", cwd=clone)
    assert info["current_repo"]["repo"] == REPO_KEY
    assert info["current_repo"]["default_pot_id"] is None

    used = _cli(
        "--json",
        "pot",
        "use",
        "managed:pot_shop",
        "--also-default-for-current-repo",
        cwd=clone,
    )
    assert (used["id"], used["origin"], used["repo_default_set"]) == (
        "pot_shop",
        "managed",
        True,
    )
    assert served.state.active["alice"] == "pot_shop"
    assert served.state.repo_defaults[("alice", REPO_KEY)] == "pot_shop"

    info = _cli("--json", "pot", "info", cwd=clone)
    assert info["current_repo"]["default_pot_id"] == "pot_shop"
    assert info["active_pot"] == {"id": "pot_shop", "name": "shop"}

    assert _cli("--json", "host", "use", "local", cwd=clone) == {
        "active_origin": "local"
    }
    local_info = _cli("--json", "pot", "info", cwd=clone)
    assert local_info["current_repo"]["default_pot_id"] != "pot_shop"
    assert _cli("--json", "host", "use", "managed", cwd=clone) == {
        "active_origin": "managed"
    }
    assert (
        _cli("--json", "pot", "info", cwd=clone)["current_repo"]["default_pot_id"]
        == "pot_shop"
    )

    # The key travelled as a bearer header, never on argv, and only path
    # under the base URL the CLI used is the typed operations route.
    assert {request["path"] for request in served.requests} == {"/graph/v1/operations"}
    assert {request["authorization"] for request in served.requests} == {
        f"Bearer {ALICE_KEY}"
    }


# --- in-process: engine paths and authorization --------------------------------


def _as(served: ServedReferenceHost, key: str) -> None:
    host_registry.set_managed_endpoint(served.base_url, key)
    host_registry.set_persisted_origin(host_registry.MANAGED)
    remote._connections.clear()
    _common._state["managed_pots"] = None


def _run(*args: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(app, ["--json", *args])
    try:
        return result.exit_code, json.loads(result.output)
    except json.JSONDecodeError:  # pragma: no cover - shown on failure
        raise AssertionError(result.output) from None


def test_engine_commands_reach_the_managed_host(served, clone) -> None:
    _as(served, ALICE_KEY)
    assert (
        _run("pot", "use", "managed:pot_shop", "--also-default-for-current-repo")[0]
        == 0
    )

    code, resolved = _run("resolve", "how do we ship")
    assert code == 0, resolved
    assert resolved["pot_id"] == "pot_shop"
    code, recorded = _run(
        "record", "--type", "workflow", "--summary", "ship with the release train"
    )
    assert code == 0, recorded
    assert recorded["accepted"] is True
    assert ("record", "pot_shop") in served.host.agent_context.calls

    code, whoami = _run("whoami")
    assert (code, whoami["subject"]) == (0, "alice")
    code, status = _run("status")
    assert code == 0, status
    assert (status["profile"], status["active_pot"]) == ("managed", "shop")

    # Selection happened on the host: the repository selector for the clone's
    # remote, never a local path.
    operations = [request["operation"] for request in served.requests]
    assert "resolve" in operations and "record" in operations


def test_a_reader_reads_but_cannot_write(served, clone) -> None:
    _as(served, BOB_KEY)

    code, _ = _run("resolve", "how do we ship", "--pot", "managed:pot_shop")
    assert code == 0
    code, refused = _run(
        "record",
        "--type",
        "workflow",
        "--summary",
        "readers cannot write this",
        "--pot",
        "managed:shop",
    )
    assert code == _common.EXIT_AUTH, refused
    assert refused["code"] == "role_insufficient"


def test_a_pot_without_a_grant_does_not_exist(served, clone) -> None:
    _as(served, CAROL_KEY)

    code, refused = _run("pot", "use", "managed:pot_shop")

    assert code == _common.EXIT_VALIDATION, refused
    assert refused["code"] == "pot_not_found"
    assert host_registry.persisted_origin() == host_registry.MANAGED
    assert "carol" not in served.state.active


def test_a_refused_key_is_an_authentication_failure(served, clone) -> None:
    _as(served, "ref_wrong_" + "z" * 32)

    code, refused = _run("pot", "use", "managed:pot_shop")

    assert code == _common.EXIT_AUTH, refused
    assert refused["code"] == "authentication_failed"


# --- the explorer: the local daemon proxies the managed host --------------------


def test_the_explorer_reads_a_managed_pot_through_the_local_daemon(
    served, clone, tmp_path, monkeypatch
) -> None:
    """``potpie ui --pot managed:<id>`` names the host in the URL, and the local
    daemon answers the explorer API for it with the key it holds."""
    import httpx

    from potpie.daemon.discovery import read_daemon_credential
    from potpie.daemon.lifecycle import Daemon

    runtime_home = tmp_path / "runtime"
    for name, value in {
        "CONTEXT_ENGINE_HOME": runtime_home,
        "XDG_CONFIG_HOME": tmp_path / "xdg",
    }.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "embedded")
    _common.set_runtime(None)

    _cli(
        "--json",
        "host",
        "set",
        served.base_url,
        "--no-check",
        "--token",
        "-",
        cwd=clone,
        stdin=ALICE_KEY + "\n",
    )
    _cli("--json", "pot", "use", "managed:pot_shop", cwd=clone)
    try:
        opened = _cli("--json", "ui", "--no-open", "--pot", "managed:shop", cwd=clone)
        assert opened["origin"] == "managed"
        assert opened["pot_id"] == "pot_shop"
        assert "host=managed" in opened["url"] and "pot=pot_shop" in opened["url"]
        assert opened["warning"] is None
        assert ALICE_KEY not in json.dumps(opened)

        base = opened["url"].split("/ui?", 1)[0]
        headers = {"Authorization": f"Bearer {read_daemon_credential(runtime_home)}"}
        pots = httpx.get(
            f"{base}/ui/api/pots", params={"host": "managed"}, headers=headers
        ).json()
        graph = httpx.get(
            f"{base}/ui/api/graph",
            params={"host": "managed", "pot": "pot_shop"},
            headers=headers,
        )
        commits = httpx.get(
            f"{base}/ui/api/commits",
            params={"host": "managed", "pot": "pot_shop"},
            headers=headers,
        )

        assert {p["id"] for p in pots["pots"] if p["origin"] == "managed"} == {
            "pot_shop",
            "pot_lab",
        }
        assert graph.status_code == 200, graph.text
        assert graph.json()["pot_id"] == "pot_shop"
        assert commits.status_code == 200, commits.text
        assert {"graph_slice", "commits"} <= {
            request["operation"] for request in served.requests
        }
    finally:
        Daemon(home=runtime_home, in_process=False).stop()
