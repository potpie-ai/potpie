"""Real process-per-command protocol ingestion and partial bulk recovery."""

# Pytest assertions are the test contract.
# ruff: noqa: S101

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from potpie_context_engine.testing import write_import_directory

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "potpie/context-engine/tests/fixtures/protocols"
FIXTURE = json.loads((FIXTURES / "fixture.json").read_text())


@pytest.fixture
def cli(tmp_path):
    env = {
        **os.environ,
        "CONTEXT_ENGINE_HOME": str(tmp_path / "state"),
        "POTPIE_HARNESS_HOME": str(tmp_path / "harness"),
        "CONTEXT_ENGINE_HOST_MODE": "in_process",
        "CONTEXT_ENGINE_BACKEND": "embedded",
        "CONTEXT_ENGINE_PROTOCOLS_ENABLED": "true",
        "CONTEXT_ENGINE_EMBEDDER": "none",
        "POTPIE_TELEMETRY_DISABLED": "1",
        "PYTHONPATH": os.pathsep.join(
            map(
                str,
                (
                    ROOT,
                    ROOT / "potpie/context-core/src",
                    ROOT / "potpie/context-engine/src",
                ),
            )
        ),
    }

    def run(*args, code=0, human=False, envelope=False):
        command = [
            sys.executable,
            "-c",
            "from potpie.cli.main import main; main()",
            "--host",
            "local",
        ]
        if not human:
            command.append("--json")
        result = subprocess.run(  # noqa: S603 - fixed CLI and local fixture arguments
            command + list(args),
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == code, (args, result.stdout, result.stderr)
        if human:
            return result.stdout
        body = json.loads(result.stdout)
        return body if envelope else body.get("result") or body

    return run


def test_protocol_bulk_resume_and_cli_projection(cli, tmp_path):
    pot = cli("pot", "create", "protocol-cli")
    selector = ("--pot", "local:protocol-cli")
    installed = cli(
        "skills",
        "install",
        "--agent",
        "codex",
        "--scope",
        "project",
        "--path",
        str(tmp_path),
    )
    assert "potpie-graph" in installed["changed"]
    reference = tmp_path / ".agents/skills/potpie-graph/references/protocols.md"
    assert "protocols.message_context" in reference.read_text()
    cli(
        "skills",
        "update",
        "potpie-graph",
        "--agent",
        "codex",
        "--scope",
        "project",
        "--path",
        str(tmp_path),
    )

    for filename, key in (
        ("demo.md", "source_ref"),
        ("modbus-03.md", "modbus_source_ref"),
    ):
        source = (FIXTURES / filename).read_text()
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
            *selector,
        )
        fetched = cli("resource", "get", FIXTURE[key], *selector)
        assert fetched["chunks"][0]["text"] == source

    payload = {"operations": copy.deepcopy(FIXTURE["operations"])}
    # An unsupported predicate fails chunk 2 after chunk 1 has committed.
    good = copy.deepcopy(payload)
    payload["operations"][1]["predicate"] = "UNSUPPORTED_PROTOCOL_RELATION"
    mutation = tmp_path / "mutation.json"
    mutation.write_text(json.dumps(payload))
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
        *selector,
        code=1,
    )
    assert partial["error"]["code"] == "unknown_predicate"
    partial = partial["error"]["detail"]
    assert partial["chunks_committed"] == 1
    assert partial["chunks"][1]["status"] == "proposal_failed"
    assert partial["chunks"][0]["commit"]["verification"]["ok"]
    assert json.loads(manifest.read_text())["chunks_committed"] == 1

    mutation.write_text(json.dumps(good))
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
        *selector,
    )
    assert resumed["status"] == "committed"
    assert resumed["chunks_committed"] == len(good["operations"]) - 1
    assert resumed["chunks"][0]["status"] == "skipped"
    assert all(chunk["commit"]["verification"]["ok"] for chunk in resumed["chunks"][1:])
    for command in ("resolve", "search"):
        discovered = cli(command, "status", "--include", "protocols", *selector)
        assert discovered["items"]
        assert any(item["include"] == "protocols" for item in discovered["items"])
    read_args = (
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
        *selector,
    )
    wire = cli(*read_args, envelope=True)
    assert wire["pot_id"] == pot["id"]
    result = wire["result"]
    item = result["items"][0]
    assert item["coverage"]["status"] == "complete"
    assert [f["path"] for f in item["fields"]] == ["header.Status", "payload.status"]
    assert [
        (type(v["raw_value"]), v["raw_value"])
        for v in item["fields"][0]["allowed_values"]
    ] == [
        (int, 0),
        (int, 2),
        (str, "2"),
        (bool, False),
    ]
    human = cli(*read_args, human=True)
    assert human.index("header.Status") < human.index("payload.status")
    assert "complete" in human and "BUSY" in human
    invalid = cli(*read_args, "--environment", "prod", code=1)
    assert not invalid["ok"]
    assert "unsupported" in json.dumps(
        invalid
    ).lower() or "does not support" in json.dumps(invalid)
