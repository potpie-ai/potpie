"""The opt-in protocol ontology through the real CLI, one process per command.

Protocols are off until ``potpie config set graph.protocols on``: before that
the include family and the view are refused. The setting is read when each
process composes its runtime, so the next command sees it. Turning it off
again hides the view without losing the data.
"""

# ruff: noqa: S101 - pytest assertions are the test contract.

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from potpie_context_engine.testing import write_import_directory

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "potpie/context-engine/tests/fixtures/protocols"
FIXTURE = json.loads((FIXTURES / "fixture.json").read_text(encoding="utf-8"))
POT = "protocol-cli"
SELECTOR = ("--pot", POT)


@pytest.fixture
def cli(tmp_path):
    env = {
        **os.environ,
        # Every home this CLI could write to is a scratch directory.
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "CONTEXT_ENGINE_HOME": str(tmp_path / "state"),
        "POTPIE_HARNESS_HOME": str(tmp_path / "harness"),
        "CONTEXT_ENGINE_HOST_MODE": "in_process",
        "CONTEXT_ENGINE_BACKEND": "embedded",
        "CONTEXT_ENGINE_EMBEDDER": "none",
        "POTPIE_TELEMETRY_DISABLED": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
    }

    def run(*args, code=0, human=False, envelope=False):
        command = [sys.executable, "-c", "from potpie.cli.main import main; main()"]
        if not human:
            command.append("--json")
        result = subprocess.run(  # noqa: S603 - fixed CLI and local fixture arguments
            command + list(args),
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == code, (args, result.stdout, result.stderr)
        if human:
            return result.stdout
        body = json.loads(result.stdout)
        return body if envelope else body.get("result") or body

    return run


def _read_args(*extra: str) -> tuple[str, ...]:
    return (
        "graph",
        "read",
        "--subgraph",
        "protocols",
        "--view",
        "message_context",
        "--scope",
        "anchor_entity_key:" + FIXTURE["names"]["request"],
        "--detail",
        "full",
        *SELECTOR,
        *extra,
    )


def _assert_protocols_absent(cli) -> None:
    catalog = cli("graph", "catalog", "--profile", "full", *SELECTOR)
    assert not catalog.get("extensions")
    views = [view.get("name") for view in catalog.get("views", ())]
    assert "protocols.message_context" not in views
    refused = cli("resolve", "status", "--include", "protocols", *SELECTOR, code=1)
    assert "Unknown include families" in json.dumps(refused)
    cli(*_read_args(), code=1)


def test_protocols_stay_off_until_the_config_key_turns_them_on(cli, tmp_path):
    pot = cli("pot", "create", POT)
    _assert_protocols_absent(cli)

    enabled = cli("config", "set", "graph.protocols", "on")
    assert enabled["value"] == "on" and enabled["restart_required"] is True
    catalog = cli("graph", "catalog", "--profile", "full", *SELECTOR)
    assert catalog["extensions"] == {"protocols": "1"}

    for filename, key in (
        ("demo.md", "source_ref"),
        ("modbus-03.md", "modbus_source_ref"),
    ):
        source = (FIXTURES / filename).read_text(encoding="utf-8")
        directory = write_import_directory(
            tmp_path / filename,
            [
                {
                    "slug": "contract",
                    "title": filename,
                    "summary": "Protocol source",
                    "ordinal": 0,
                    "content_hash": hashlib.sha256(source.encode()).hexdigest(),
                    "chunks": [{"label": "Contract", "text": source}],
                }
            ],
        )
        cli(
            "resource",
            "import",
            str(directory),
            "--doc",
            FIXTURE[key].split("/")[3],
            *SELECTOR,
        )
        fetched = cli("resource", "get", FIXTURE[key], *SELECTOR)
        assert fetched["chunks"][0]["text"] == source

    # An unsupported predicate fails chunk 2 after chunk 1 has committed.
    good = {"operations": copy.deepcopy(FIXTURE["operations"])}
    payload = copy.deepcopy(good)
    payload["operations"][1]["predicate"] = "UNSUPPORTED_PROTOCOL_RELATION"
    mutation = tmp_path / "mutation.json"
    mutation.write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "partial.json"
    partial = cli(
        "graph",
        "bulk",
        "apply",
        "--file",
        str(mutation),
        "--chunk-size",
        "1",
        "--verify",
        "--manifest",
        str(manifest),
        *SELECTOR,
        code=1,
    )
    partial = partial["error"]["detail"]
    assert partial["chunks_committed"] == 1
    assert partial["chunks"][1]["status"] == "proposal_failed"
    assert partial["chunks"][0]["commit"]["verification"]["ok"]
    assert json.loads(manifest.read_text())["chunks_committed"] == 1

    mutation.write_text(json.dumps(good), encoding="utf-8")
    resumed = cli(
        "graph",
        "bulk",
        "apply",
        "--file",
        str(mutation),
        "--chunk-size",
        "1",
        "--start-chunk",
        "2",
        "--verify",
        *SELECTOR,
    )
    assert resumed["status"] == "committed"
    assert resumed["chunks_committed"] == len(good["operations"]) - 1
    assert all(chunk["commit"]["verification"]["ok"] for chunk in resumed["chunks"][1:])

    for command in ("resolve", "search"):
        discovered = cli(command, "status", "--include", "protocols", *SELECTOR)
        assert any(item["include"] == "protocols" for item in discovered["items"])

    wire = cli(*_read_args(), envelope=True)
    assert wire["pot_id"] == pot["id"]
    item = wire["result"]["items"][0]
    assert item["coverage"]["status"] == "complete"
    assert [f["path"] for f in item["fields"]] == ["header.Status", "payload.status"]
    assert [
        (type(v["raw_value"]), v["raw_value"])
        for v in item["fields"][0]["allowed_values"]
    ] == [(int, 0), (int, 2), (str, "2"), (bool, False)]
    assert not wire["result"].get("output_budget")
    human = cli(*_read_args(), human=True)
    assert human.index("header.Status") < human.index("payload.status")
    assert "complete" in human and "BUSY" in human
    cli(*_read_args("--environment", "prod"), code=1)

    # Off again: the view is gone, the data is not.
    cli("config", "set", "graph.protocols", "off")
    _assert_protocols_absent(cli)
    cli("config", "set", "graph.protocols", "on")
    again = cli(*_read_args())
    assert again["items"][0]["entity_key"] == FIXTURE["names"]["request"]
    assert again["items"][0]["coverage"]["status"] == "complete"
