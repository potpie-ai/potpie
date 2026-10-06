"""Embedding hosts take the pot and auth DTOs from root ``potpie``.

``PotInfo``, ``SourceInfo``, ``PotAggregateStatus`` and ``AuthIdentity`` are the
control-plane values a hosted context service exchanges with the CLI. They stay
in the product distribution; importing them must not pull in the daemon, the
CLI or the local runtime composition, and must not touch the Potpie home, so a
host can use them without a daemon ever starting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_SNIPPET = """
import json
import sys

from potpie.auth.ports.identity import AuthIdentity
from potpie.pots.contracts import PotAggregateStatus, PotInfo, SourceInfo

status = PotAggregateStatus(
    active_pot=PotInfo(pot_id="pot_demo", name="demo", active=True),
    pot_count=1,
    sources=(SourceInfo(source_id="src_demo", kind="repo", name="demo"),),
    backend_ready=True,
)
identity = AuthIdentity(subject="service-user:alice", mode="managed")
assert status.active_pot.pot_id == "pot_demo" and identity.mode == "managed"
print(json.dumps(sorted(sys.modules)))
"""


def test_pot_and_auth_dtos_import_without_daemon_cli_or_runtime(tmp_path) -> None:
    home = tmp_path / "potpie-home"
    env = {**os.environ, "CONTEXT_ENGINE_HOME": str(home), "HOME": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    product_modules = {
        name for name in loaded if name == "potpie" or name.startswith("potpie.")
    }
    assert not {
        name
        for name in product_modules
        if name.startswith(("potpie.daemon", "potpie.cli", "potpie.runtime"))
    }
    assert not {name.split(".")[0] for name in loaded} & {
        "fastapi",
        "typer",
        "uvicorn",
        "keyring",
    }
    assert not home.exists()
