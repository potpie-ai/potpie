from __future__ import annotations

from pathlib import Path

from domain.ports.daemon.lifecycle import (
    DaemonDiscovery,
    DaemonStartResult,
    DaemonStatus,
)
from potpie.daemon.process.ipc_client import parse_discovery


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_ENGINE = ROOT / "potpie" / "context-engine"


def _python_sources(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        and "__pycache__" not in path.parts
    ]


def test_context_engine_source_has_no_product_telemetry_imports() -> None:
    forbidden = (
        "sentry_sdk",
        "bootstrap.sentry_",
        "potpie.runtime.telemetry.sentry",
        "potpie.cli.telemetry.product_analytics",
        "potpie.cli.telemetry.sentry",
    )

    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_sources(CONTEXT_ENGINE)
        if any(token in path.read_text(encoding="utf-8") for token in forbidden)
    ]

    assert offenders == []


def test_context_engine_source_has_no_root_cli_or_daemon_imports() -> None:
    forbidden = (
        "potpie.cli",
        "potpie.daemon",
        "adapters.inbound.cli",
        "adapters.outbound." + "cli_auth",
        "domain.ports." + "cli_auth",
        "bootstrap." + "cli_auth_wiring",
        "host.daemon",
    )

    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_sources(CONTEXT_ENGINE)
        if any(token in path.read_text(encoding="utf-8") for token in forbidden)
    ]

    assert offenders == []


def test_context_engine_metadata_has_no_cli_auth_dependencies() -> None:
    pyproject = (CONTEXT_ENGINE / "pyproject.toml").read_text(encoding="utf-8")

    assert "keyring" not in pyproject
    assert "cli_auth_e2e" not in pyproject


def test_root_runtime_imports_context_engine_env_bootstrap_only_from_wrapper() -> None:
    forbidden = (
        "bootstrap._build_info",
        "potpie.cli.auth.credentials_store",
    )

    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_sources(ROOT / "potpie" / "runtime")
        if any(token in path.read_text(encoding="utf-8") for token in forbidden)
    ]

    assert offenders == []

    bootstrap_import_offenders = [
        str(path.relative_to(ROOT))
        for path in _python_sources(ROOT / "potpie" / "runtime")
        if path.name != "env_bootstrap.py"
        and "bootstrap.env_bootstrap" in path.read_text(encoding="utf-8")
    ]

    assert bootstrap_import_offenders == []


def test_root_daemon_rpc_does_not_own_engine_contract_tables() -> None:
    root_rpc = (ROOT / "potpie" / "daemon" / "rpc.py").read_text(encoding="utf-8")
    engine_contract = (
        CONTEXT_ENGINE / "domain" / "ports" / "daemon" / "rpc_contract.py"
    ).read_text(encoding="utf-8")

    assert "_ALLOWED_RPC_CLASS_REFS" not in root_rpc
    assert "RPC_DTO_MODULES" not in root_rpc
    assert "RPC_SURFACES: Mapping" not in root_rpc
    assert "domain.actor" not in root_rpc
    assert "RPC_DTO_MODULES" in engine_contract
    assert "RPC_SURFACES: Mapping" in engine_contract


def test_root_and_context_engine_env_bootstrap_core_behavior_stays_in_sync() -> None:
    from bootstrap import env_bootstrap as engine_env_bootstrap
    from potpie.runtime import env_bootstrap as root_env_bootstrap

    lines = [
        "",
        "# comment",
        "PLAIN=value",
        " export EXPORTED = quoted ",
        "SINGLE='one two'",
        'DOUBLE="three four"',
        "NO_EQUALS",
        "=missing_key",
    ]

    assert root_env_bootstrap._PROJECT_ROOT_MARKERS == (
        engine_env_bootstrap._PROJECT_ROOT_MARKERS
    )
    assert [
        root_env_bootstrap._parse_env_line(line) for line in lines
    ] == [engine_env_bootstrap._parse_env_line(line) for line in lines]


def test_public_daemon_lifecycle_uses_active_http_discovery_contract() -> None:
    assert "bind" not in DaemonDiscovery.__annotations__
    assert "socket" not in DaemonStatus.__annotations__
    assert "socket" not in DaemonStartResult.__annotations__

    discovery = parse_discovery(
        {
            "transport": "http",
            "base_url": "http://127.0.0.1:4321",
            "token": "secret",
            "pid": "123",
            "bind": "unix:/tmp/legacy.sock",
        }
    )

    assert discovery == {
        "transport": "http",
        "base_url": "http://127.0.0.1:4321",
        "token": "secret",
        "pid": 123,
    }


def test_public_cli_daemon_paths_do_not_use_legacy_operation_client() -> None:
    active_paths = (
        ROOT / "potpie" / "cli" / "commands" / "service.py",
        ROOT / "potpie" / "daemon" / "lifecycle.py",
        ROOT / "potpie" / "daemon" / "process" / "launcher.py",
    )

    offenders = [
        str(path.relative_to(ROOT))
        for path in active_paths
        if "legacy_client_for" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
