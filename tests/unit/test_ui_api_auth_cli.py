"""``potpie ui`` is the CLI end of the browser handoff.

It is the one party that can read the daemon token from the owner-only
credential file, so it is the one that spends the token: on the staleness
probe, and on the single-use code the opened URL carries. What the daemon must
never receive is an anonymous request, and what the browser (or whoever runs
``potpie ui --no-open --json``) must never receive is the token itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.commands import ui as ui_cmd
from potpie.daemon.discovery import write_daemon_credential

pytestmark = pytest.mark.unit

TOKEN = "daemon-token-for-tests-0123456789abcdef"  # noqa: S105 - test fixture
BASE = "http://127.0.0.1:8765"
BEARER = {"Authorization": f"Bearer {TOKEN}"}


class _Response:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class _Daemon:
    def __init__(self, home: Path) -> None:
        self.home = home

    def ensure(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"ready": True, "url": BASE}


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)
    _common.set_runtime(None)


@pytest.fixture
def daemon(tmp_path: Path, monkeypatch) -> _Daemon:
    write_daemon_credential(tmp_path, TOKEN)
    fake = _Daemon(tmp_path)
    monkeypatch.setattr(ui_cmd, "get_root_runtime", lambda: object())
    monkeypatch.setattr(ui_cmd, "get_daemon_service", lambda runtime: fake)
    return fake


@pytest.fixture
def calls(daemon, monkeypatch) -> dict[str, tuple[str, dict[str, str]]]:
    """A daemon that answers both handoff steps, recording what it was sent."""
    import httpx

    seen: dict[str, tuple[str, dict[str, str]]] = {}

    def _get(url: str, headers: Any = None, timeout: float | None = None) -> _Response:
        seen["get"] = (url, dict(headers or {}))
        return _Response(200, {"pots": [], "active": None})

    def _post(url: str, headers: Any = None, timeout: float | None = None) -> _Response:
        seen["post"] = (url, dict(headers or {}))
        return _Response(200, {"code": "handoff-code", "expires_in": 120})

    monkeypatch.setattr(httpx, "get", _get)
    monkeypatch.setattr(httpx, "post", _post)
    return seen


def _run_json(*args: str) -> dict[str, Any]:
    _common.set_json(True)
    app = typer.Typer()
    ui_cmd.register(app)
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_the_opened_url_carries_a_handoff_code_not_the_token(calls, capsys) -> None:
    ui_cmd.ui_command(open_browser=False, pot=None)

    printed = capsys.readouterr().out
    assert f"{BASE}/ui?k=handoff-code" in printed
    assert TOKEN not in printed
    # Both daemon calls are authenticated: an anonymous probe would report a
    # working daemon as broken, and an anonymous mint would be refused.
    assert calls["get"] == (f"{BASE}/ui/api/pots", BEARER)
    assert calls["post"] == (f"{BASE}/ui/api/handoff", BEARER)


def test_no_open_json_keeps_the_url_and_warning_contract(calls) -> None:
    """Callers that open the page themselves read ``url`` and ``warning``. The
    URL must carry the code (the page is useless without a session) and must
    never carry the daemon token."""
    payload = _run_json("--no-open")

    assert payload == {
        "url": f"{BASE}/ui?k=handoff-code",
        "pot_id": None,
        "warning": None,
    }
    assert TOKEN not in json.dumps(payload)


def test_the_browser_is_opened_on_the_handoff_url(calls, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(ui_cmd.webbrowser, "open", opened.append)

    ui_cmd.ui_command(open_browser=True, pot=None)

    assert opened == [f"{BASE}/ui?k=handoff-code"]


def test_a_daemon_that_rejects_the_credential_is_reported(calls, monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(401))

    payload = _run_json("--no-open")

    assert "rejected this machine's daemon credential" in payload["warning"]
    assert "k=" not in payload["url"]


def test_an_unreadable_credential_sends_no_header_and_is_reported(
    calls, daemon, monkeypatch
) -> None:
    import httpx

    (daemon.home / "daemon.credential").unlink()
    seen: list[dict[str, str]] = []

    def _get(url: str, headers: Any = None, timeout: float | None = None):
        seen.append(dict(headers or {}))
        return _Response(401)

    monkeypatch.setattr(httpx, "get", _get)

    payload = _run_json("--no-open")

    assert seen == [{}]
    assert payload["warning"]
    assert "k=" not in payload["url"]


def test_a_daemon_that_refuses_a_session_is_reported(
    calls, monkeypatch, capsys
) -> None:
    """Without this the explorer opens onto a wall of 401s with no account of
    why, which reads as "the graph is broken" rather than "restart the daemon"."""
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response(403))

    ui_cmd.ui_command(open_browser=False, pot=None)

    printed = capsys.readouterr().out
    assert "would not issue a browser session" in printed
    # No code, so nothing pretends the browser will be able to read anything.
    assert "k=" not in printed


def test_a_daemon_from_before_the_gate_still_opens(calls, monkeypatch, capsys) -> None:
    """A daemon that has never heard of the handoff route serves the API open,
    so the page works without a code; warning about it would be noise."""
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response(404))

    ui_cmd.ui_command(open_browser=False, pot=None)

    printed = capsys.readouterr().out
    assert f"{BASE}/ui" in printed
    assert "k=" not in printed
    assert "!" not in printed


# --- coming back without a code ---------------------------------------------
#
# 404 above is the only silence the user can act on being told nothing about.
# In every case below the command would otherwise print the URL, say "(opening
# in your browser…)", and open a page whose every API call 401s.


def _mint_returns(monkeypatch, response_or_error) -> None:
    import httpx

    def _post(*args, **kwargs):
        if isinstance(response_or_error, Exception):
            raise response_or_error
        return response_or_error

    monkeypatch.setattr(httpx, "post", _post)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_Response(200, {"expires_in": 120}), "returned no session code"),
        (_Response(200, {"code": ""}), "returned no session code"),
        # `_Response.json()` raises with no payload: a body this command cannot
        # parse leaves it just as empty-handed as one that carried nothing.
        (_Response(200), "returned no session code"),
        (
            ConnectionError("all connection attempts failed"),
            "could not reach the daemon",
        ),
    ],
    ids=["no-code-key", "empty-code", "unparseable-body", "unreachable"],
)
def test_a_mint_that_produces_nothing_is_never_reported_as_success(
    calls, monkeypatch, capsys, outcome, expected: str
) -> None:
    _mint_returns(monkeypatch, outcome)

    ui_cmd.ui_command(open_browser=False, pot=None)

    printed = capsys.readouterr().out
    assert expected in printed
    assert "k=" not in printed
    assert "opening in your browser" not in printed


def test_a_mint_that_produces_nothing_does_not_open_a_dead_page(
    calls, monkeypatch
) -> None:
    """Opening the browser on a session that was never issued spends the user's
    attention on a page that can only show 401s."""
    opened: list[str] = []
    monkeypatch.setattr(ui_cmd.webbrowser, "open", opened.append)
    _mint_returns(monkeypatch, _Response(200, {"expires_in": 120}))

    ui_cmd.ui_command(open_browser=True, pot=None)

    assert opened == []
