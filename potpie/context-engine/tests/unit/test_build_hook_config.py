from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import build_config_values


def _load_distribution_defaults_hook(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
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
    path = Path(__file__).resolve().parents[2] / "distribution_defaults_hook.py"
    spec = importlib.util.spec_from_file_location(
        "_test_distribution_defaults_hook",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hide_build_env(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Shadow ambient .env values while still treating the build input as missing."""
    for name in names:
        monkeypatch.setenv(name, " ")


def test_distribution_defaults_use_internal_field_names() -> None:
    values = build_config_values.distribution_default_values(
        {
            "POTPIE_ENVIRONMENT": "prod_oss",
            "POTPIE_SENTRY_DSN": "https://sentry.example.invalid/1",
            "POTPIE_POSTHOG_API_KEY": "phc_public",
            "POTPIE_POSTHOG_HOST": "https://posthog.invalid",
            "LINEAR_CLIENT_ID": "linear-client",
            "POTPIE_GITHUB_CLIENT_ID": "github-client",
        }
    )

    assert values == {
        "environment": "prod_oss",
        "sentry_dsn": "https://sentry.example.invalid/1",
        "posthog_api_key": "phc_public",
        "posthog_host": "https://posthog.invalid",
        "linear_client_id": "linear-client",
        "github_client_id": "github-client",
    }


def test_distribution_default_input_mapping_is_exhaustive() -> None:
    fields = set(build_config_values.distribution_default_values({}))

    assert set(build_config_values.DISTRIBUTION_DEFAULT_INPUT_NAMES_BY_FIELD) == fields


def test_distribution_defaults_load_nearest_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "POTPIE_ENVIRONMENT=prod_oss",
                "POTPIE_SENTRY_DSN=file-sentry",
                "POTPIE_POSTHOG_API_KEY=file-posthog",
                "POTPIE_POSTHOG_HOST=https://posthog.invalid",
                "LINEAR_CLIENT_ID=file-linear",
                "export POTPIE_GITHUB_CLIENT_ID='file-github'",
            ]
        ),
        encoding="utf-8",
    )
    nested = tmp_path / "potpie" / "context-engine"
    nested.mkdir(parents=True)

    values = build_config_values.distribution_default_values({}, dotenv_start=nested)

    assert values == {
        "environment": "prod_oss",
        "sentry_dsn": "file-sentry",
        "posthog_api_key": "file-posthog",
        "posthog_host": "https://posthog.invalid",
        "linear_client_id": "file-linear",
        "github_client_id": "file-github",
    }


def test_build_config_environ_overrides_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "POTPIE_SENTRY_DSN=file-sentry",
                "POTPIE_POSTHOG_API_KEY=file-posthog",
                "LINEAR_CLIENT_ID=file-linear",
                "POTPIE_GITHUB_CLIENT_ID=file-github",
            ]
        ),
        encoding="utf-8",
    )

    values = build_config_values.distribution_default_values(
        {
            "POTPIE_SENTRY_DSN": "env-sentry",
            "POTPIE_POSTHOG_API_KEY": "env-posthog",
            "LINEAR_CLIENT_ID": "env-linear",
            "POTPIE_GITHUB_CLIENT_ID": "env-github",
        },
        dotenv_start=tmp_path,
    )

    assert values["sentry_dsn"] == "env-sentry"
    assert values["posthog_api_key"] == "env-posthog"
    assert values["linear_client_id"] == "env-linear"
    assert values["github_client_id"] == "env-github"


def test_build_config_input_detection(tmp_path: Path) -> None:
    assert not build_config_values.has_build_config_inputs(
        build_config_values.DISTRIBUTION_DEFAULT_INPUT_NAMES,
        {},
        dotenv_start=tmp_path,
    )

    assert build_config_values.has_build_config_inputs(
        build_config_values.DISTRIBUTION_DEFAULT_INPUT_NAMES,
        {"LINEAR_CLIENT_ID": ""},
        dotenv_start=tmp_path,
    )

    (tmp_path / ".env").write_text("UNRELATED=1\n", encoding="utf-8")
    assert not build_config_values.has_build_config_inputs(
        build_config_values.DISTRIBUTION_DEFAULT_INPUT_NAMES,
        {},
        dotenv_start=tmp_path,
    )

    (tmp_path / ".env").write_text("POTPIE_SENTRY_DSN=file-sentry\n", encoding="utf-8")
    assert build_config_values.has_build_config_inputs(
        build_config_values.DISTRIBUTION_DEFAULT_INPUT_NAMES,
        {},
        dotenv_start=tmp_path,
    )


def test_distribution_defaults_have_safe_public_defaults() -> None:
    values = build_config_values.distribution_default_values({})

    assert values["environment"] == "prod_oss"
    assert values["posthog_host"] == "https://us.i.posthog.com"
    assert values["sentry_dsn"] == ""
    assert values["posthog_api_key"] == ""


def test_build_info_defaults_git_sha_to_github_sha() -> None:
    values = build_config_values.build_info_values(
        {"GITHUB_SHA": "abc123", "POTPIE_BUILD_TIME": "2026-06-28T00:00:00Z"}
    )

    assert values == {"GIT_SHA": "abc123", "BUILD_TIME": "2026-06-28T00:00:00Z"}


def test_write_distribution_mapping_does_not_emit_env_var_constants(tmp_path) -> None:
    out = tmp_path / "_distribution_defaults.py"

    build_config_values.write_python_mapping(
        out,
        "DISTRIBUTION_DEFAULTS",
        {"environment": "prod_oss", "sentry_dsn": ""},
    )

    text = out.read_text(encoding="utf-8")
    assert "DISTRIBUTION_DEFAULTS = {" in text
    assert "'environment': 'prod_oss'" in text
    assert "POTPIE_SENTRY_DSN =" not in text
    assert "LINEAR_CLIENT_ID =" not in text


def test_write_python_constants(tmp_path: Path) -> None:
    out = tmp_path / "_build_info.py"

    build_config_values.write_python_constants(out, {"GIT_SHA": "abc123"})

    assert out.read_text(encoding="utf-8").splitlines() == [
        "# Auto-generated at wheel build time - do not edit manually.",
        "# Runtime environment variables override these packaged public defaults.",
        "GIT_SHA = 'abc123'",
    ]


def test_prefer_existing_distribution_defaults_preserves_missing_field_inputs(
    tmp_path: Path,
) -> None:
    out = tmp_path / "_distribution_defaults.py"
    build_config_values.write_python_mapping(
        out,
        "DISTRIBUTION_DEFAULTS",
        {
            "environment": "prod_oss",
            "sentry_dsn": "old-sentry",
            "posthog_api_key": "old-posthog",
            "posthog_host": "https://old.posthog.invalid",
            "linear_client_id": "old-linear",
            "github_client_id": "old-github",
        },
    )

    values = build_config_values.prefer_existing_distribution_default_values(
        out,
        build_config_values.distribution_default_values(
            {"POTPIE_ENVIRONMENT": "staging", "POTPIE_SENTRY_DSN": " "}
        ),
        environ={"POTPIE_ENVIRONMENT": "staging", "POTPIE_SENTRY_DSN": " "},
    )

    assert values == {
        "environment": "staging",
        "sentry_dsn": "old-sentry",
        "posthog_api_key": "old-posthog",
        "posthog_host": "https://old.posthog.invalid",
        "linear_client_id": "old-linear",
        "github_client_id": "old-github",
    }


def test_distribution_defaults_hook_uses_field_aware_preservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    distribution_defaults_hook = _load_distribution_defaults_hook(monkeypatch)
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    build_config_values.write_python_mapping(
        bootstrap_dir / "_distribution_defaults.py",
        "DISTRIBUTION_DEFAULTS",
        {
            "environment": "prod_oss",
            "sentry_dsn": "old-sentry",
            "posthog_api_key": "old-posthog",
            "posthog_host": "https://old.posthog.invalid",
            "linear_client_id": "old-linear",
            "github_client_id": "old-github",
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "staging")
    _hide_build_env(
        monkeypatch,
        "POTPIE_SENTRY_DSN",
        "POTPIE_POSTHOG_API_KEY",
        "POTPIE_POSTHOG_HOST",
        "LINEAR_CLIENT_ID",
        "POTPIE_GITHUB_CLIENT_ID",
    )
    monkeypatch.delenv("POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS", raising=False)
    hook = object.__new__(distribution_defaults_hook.DistributionDefaultsHook)
    build_data: dict[str, list[str]] = {}

    hook.initialize("wheel", build_data)

    generated = build_config_values._read_python_mapping(
        bootstrap_dir / "_distribution_defaults.py",
        "DISTRIBUTION_DEFAULTS",
    )
    assert generated == {
        "environment": "staging",
        "sentry_dsn": "old-sentry",
        "posthog_api_key": "old-posthog",
        "posthog_host": "https://old.posthog.invalid",
        "linear_client_id": "old-linear",
        "github_client_id": "old-github",
    }
    assert build_data == {
        "artifacts": [
            "bootstrap/_distribution_defaults.py",
            "bootstrap/_build_info.py",
        ]
    }


def test_prefer_existing_build_info(tmp_path: Path) -> None:
    out = tmp_path / "_build_info.py"
    build_config_values.write_python_constants(
        out, {"GIT_SHA": "old", "BUILD_TIME": "kept"}
    )

    values = build_config_values.prefer_existing_config_values(
        out,
        {"GIT_SHA": "", "BUILD_TIME": "default", "EXTRA": "new"},
    )

    assert values == {"GIT_SHA": "old", "BUILD_TIME": "kept", "EXTRA": "new"}


def test_release_validation_fails_when_required_defaults_are_missing() -> None:
    with pytest.raises(RuntimeError, match="sentry_dsn"):
        build_config_values.validate_distribution_defaults(
            {
                "environment": "prod_oss",
                "sentry_dsn": "",
                "posthog_api_key": "phc_public",
                "posthog_host": "https://us.i.posthog.com",
                "linear_client_id": "linear-client",
                "github_client_id": "github-client",
            }
        )


def test_release_validation_passes_when_required_defaults_are_present() -> None:
    build_config_values.validate_distribution_defaults(
        {
            "environment": "prod_oss",
            "sentry_dsn": "https://sentry.example.invalid/1",
            "posthog_api_key": "phc_public",
            "posthog_host": "https://us.i.posthog.com",
            "linear_client_id": "linear-client",
            "github_client_id": "github-client",
        }
    )


def test_local_build_without_validation_does_not_fail() -> None:
    values = build_config_values.distribution_default_values({})

    assert build_config_values.should_validate_distribution_defaults({}) is False
    assert values["sentry_dsn"] == ""


def test_explicit_validation_flag_is_required() -> None:
    assert (
        build_config_values.should_validate_distribution_defaults(
            {"POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS": "1"}
        )
        is True
    )
