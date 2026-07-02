from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import telemetry_build_config_values as build_config


def _load_telemetry_build_hook(monkeypatch) -> ModuleType:
    interface = ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = object
    monkeypatch.setitem(sys.modules, "hatchling", ModuleType("hatchling"))
    monkeypatch.setitem(
        sys.modules, "hatchling.builders", ModuleType("hatchling.builders")
    )
    monkeypatch.setitem(
        sys.modules,
        "hatchling.builders.hooks",
        ModuleType("hatchling.builders.hooks"),
    )
    monkeypatch.setitem(
        sys.modules,
        "hatchling.builders.hooks.plugin",
        ModuleType("hatchling.builders.hooks.plugin"),
    )
    monkeypatch.setitem(
        sys.modules,
        "hatchling.builders.hooks.plugin.interface",
        interface,
    )
    path = Path(__file__).resolve().parents[2] / "telemetry_build_hook.py"
    spec = importlib.util.spec_from_file_location("_test_telemetry_build_hook", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sentry_build_config_defaults() -> None:
    values = build_config.sentry_config_values({})

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "0",
        "POTPIE_SENTRY_ENABLED": "1",
        "POTPIE_SENTRY_DSN": "",
        "POTPIE_SENTRY_ENVIRONMENT": "prod_oss",
        "POTPIE_SENTRY_RELEASE": "",
        "POTPIE_SENTRY_DIST": "",
    }


def test_sentry_build_config_reads_runtime_defaults() -> None:
    values = build_config.sentry_config_values(
        {
            "POTPIE_TELEMETRY_DISABLED": "1",
            "POTPIE_SENTRY_ENABLED": "0",
            "POTPIE_SENTRY_DSN": "https://sentry.example/1",
            "POTPIE_SENTRY_ENVIRONMENT": "staging",
            "POTPIE_SENTRY_RELEASE": "potpie@test",
            "GITHUB_SHA": "abc123",
        }
    )

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "1",
        "POTPIE_SENTRY_ENABLED": "0",
        "POTPIE_SENTRY_DSN": "https://sentry.example/1",
        "POTPIE_SENTRY_ENVIRONMENT": "staging",
        "POTPIE_SENTRY_RELEASE": "potpie@test",
        "POTPIE_SENTRY_DIST": "abc123",
    }


def test_posthog_build_config_defaults() -> None:
    values = build_config.posthog_config_values({})

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "0",
        "POTPIE_POSTHOG_ENABLED": "1",
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": "1",
        "POTPIE_POSTHOG_API_KEY": "",
        "POTPIE_POSTHOG_HOST": "https://us.i.posthog.com",
    }


def test_posthog_build_config_reads_cli_defaults() -> None:
    values = build_config.posthog_config_values(
        {
            "POTPIE_TELEMETRY_DISABLED": "1",
            "POTPIE_POSTHOG_ENABLED": "0",
            "POTPIE_PRODUCT_ANALYTICS_ENABLED": "0",
            "POTPIE_POSTHOG_API_KEY": "ph_test",
            "POTPIE_POSTHOG_HOST": "https://eu.i.posthog.com",
        }
    )

    assert values == {
        "POTPIE_TELEMETRY_DISABLED": "1",
        "POTPIE_POSTHOG_ENABLED": "0",
        "POTPIE_PRODUCT_ANALYTICS_ENABLED": "0",
        "POTPIE_POSTHOG_API_KEY": "ph_test",
        "POTPIE_POSTHOG_HOST": "https://eu.i.posthog.com",
    }


def test_write_python_constants(tmp_path) -> None:
    out = tmp_path / "runtime" / "telemetry" / "_build_config.py"

    build_config.write_python_constants(out, {"A": "x", "B": ""})

    assert out.read_text(encoding="utf-8").splitlines() == [
        "# Auto-generated at wheel build time - do not edit manually.",
        "# Override any value at runtime by setting the corresponding environment variable.",
        "A = 'x'",
        "B = ''",
    ]


def test_root_telemetry_build_hook_force_includes_temp_generated_files(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("POTPIE_SENTRY_DSN", "https://sentry.example/1")
    monkeypatch.setenv("POTPIE_POSTHOG_API_KEY", "phc_test")
    hook_module = _load_telemetry_build_hook(monkeypatch)
    hook = object.__new__(hook_module.TelemetryBuildHook)
    build_data: dict[str, object] = {}

    hook.initialize("wheel", build_data)

    force_include = build_data["force_include"]
    assert isinstance(force_include, dict)
    sentry_source = _force_include_source(
        force_include,
        "potpie/runtime/telemetry/_build_config.py",
    )
    posthog_source = _force_include_source(
        force_include,
        "potpie/cli/telemetry/_build_config.py",
    )
    assert sentry_source.exists()
    assert posthog_source.exists()
    assert sentry_source != build_config.SENTRY_TELEMETRY_OUT
    assert posthog_source != build_config.POSTHOG_TELEMETRY_OUT
    assert "POTPIE_SENTRY_DSN = 'https://sentry.example/1'" in (
        sentry_source.read_text(encoding="utf-8")
    )
    assert "POTPIE_POSTHOG_API_KEY = 'phc_test'" in (
        posthog_source.read_text(encoding="utf-8")
    )

    generated_root = sentry_source.parents[3]
    hook.finalize("wheel", build_data, "dist/potpie.whl")
    assert not generated_root.exists()


def _force_include_source(
    force_include: dict[str, str],
    destination: str,
) -> Path:
    for source, target in force_include.items():
        if target == destination:
            return Path(source)
    raise AssertionError(f"Missing force_include destination: {destination}")
