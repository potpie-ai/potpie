"""One-time document configuration stays private and changes later ingests."""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

from typer.testing import CliRunner

from potpie.auth.adapters.credentials_store import load_secure_secret
from potpie.cli.commands import document
from potpie.cli.main import app
from potpie.config.local import LocalConfigService
from potpie_context_engine.adapters.outbound.resources import capabilities
from potpie_context_engine.adapters.outbound.resources.enrichers import (
    vision_httpx_ollama,
)


def _isolate_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in (
        "POTPIE_FACTS_LLM_PROVIDER",
        "POTPIE_FACTS_LLM_BASE_URL",
        "POTPIE_FACTS_LLM_MODEL",
        "POTPIE_FACTS_LLM_API_KEY",
        "POTPIE_VISION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_document_setup_saves_defaults_and_secret_without_echo(
    tmp_path, monkeypatch
) -> None:
    _isolate_home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        capabilities,
        "documents_capabilities",
        lambda: {"docling": True, "rapidocr": True, "httpx": True},
    )
    monkeypatch.setattr(vision_httpx_ollama, "ollama_reachable", lambda: True)
    monkeypatch.setattr(
        vision_httpx_ollama, "ollama_model_supports_vision", lambda model: True
    )
    monkeypatch.setattr(
        vision_httpx_ollama,
        "ollama_model_available",
        lambda model: model == "moondream",
        raising=False,
    )

    result = CliRunner().invoke(
        app, ["document", "setup"], input="y\n\ny\ntest-gemini-private-key\n"
    )

    assert result.exit_code == 0, result.output
    assert "test-gemini-private-key" not in result.output
    assert (
        "test-gemini-private-key" not in (tmp_path / "home" / "config.json").read_text()
    )
    assert load_secure_secret("document_facts_llm_api_key") == "test-gemini-private-key"
    assert LocalConfigService().get("document.vision_provider") == "local"
    assert LocalConfigService().get("document.extract_facts") == "on"
    secret_file = tmp_path / "config" / "potpie" / "integration_secrets.json"
    if os.name != "nt":
        assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600


def test_document_setup_accepts_an_installed_vision_model(tmp_path, monkeypatch) -> None:
    _isolate_home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        capabilities,
        "documents_capabilities",
        lambda: {"docling": True, "rapidocr": True, "httpx": True},
    )
    monkeypatch.setattr(vision_httpx_ollama, "ollama_reachable", lambda: True)
    monkeypatch.setattr(
        vision_httpx_ollama, "ollama_model_supports_vision", lambda model: True
    )
    monkeypatch.setattr(
        vision_httpx_ollama,
        "ollama_model_available",
        lambda model: model == "better-vision:latest",
    )

    result = CliRunner().invoke(
        app,
        ["document", "setup"],
        input="y\nbetter-vision:latest\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert LocalConfigService().get("document.vision_model") == "better-vision:latest"


def test_document_setup_rejects_installed_text_only_model(tmp_path, monkeypatch) -> None:
    _isolate_home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        capabilities,
        "documents_capabilities",
        lambda: {"docling": True, "rapidocr": True, "httpx": True},
    )
    monkeypatch.setattr(vision_httpx_ollama, "ollama_reachable", lambda: True)
    monkeypatch.setattr(
        vision_httpx_ollama, "ollama_model_available", lambda model: True
    )
    monkeypatch.setattr(
        vision_httpx_ollama, "ollama_model_supports_vision", lambda model: False
    )

    result = CliRunner().invoke(
        app, ["--json", "document", "setup"], input="y\ntext-only\n"
    )
    assert result.exit_code != 0
    assert json.loads(result.output)["code"] == "interactive_required"
    human = CliRunner().invoke(
        app, ["document", "setup"], input="y\ntext-only\n"
    )
    assert human.exit_code != 0
    assert "does not report Ollama vision capability" in human.output


def test_document_ingest_uses_saved_defaults_and_key(tmp_path, monkeypatch) -> None:
    _isolate_home(tmp_path, monkeypatch)
    config = LocalConfigService()
    config.set("document.vision_provider", "local")
    config.set("document.vision_model", "moondream")
    config.set("document.extract_facts", "on")
    config.set("document.facts_provider", "openai")
    config.set("document.facts_model", "gemini-3.5-flash-lite")
    config.set(
        "document.facts_base_url",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    from potpie.auth.adapters.credentials_store import store_secure_secret

    store_secure_secret("document_facts_llm_api_key", "saved-secret")
    seen = {}

    class Service:
        def ingest_file(self, **kwargs):
            seen.update(kwargs)
            seen["provider"] = os.environ.get("POTPIE_FACTS_LLM_PROVIDER")
            seen["model"] = os.environ.get("POTPIE_FACTS_LLM_MODEL")
            seen["base_url"] = os.environ.get("POTPIE_FACTS_LLM_BASE_URL")
            seen["key"] = os.environ.get("POTPIE_FACTS_LLM_API_KEY")
            seen["vision_model"] = os.environ.get("POTPIE_VISION_MODEL")
            return SimpleNamespace(
                graph_written=True,
                errors=[],
                doc_slug="manual",
                parser_tier="docling",
                provenance_version=2,
                sections_added=[],
                sections_kept=[],
                sections_changed=[],
                sections_removed=[],
                elements_added=[],
                elements_removed=[],
                summary_pending=[],
                facts_verified=0,
                facts_proposed=0,
                facts_suppressed=0,
                fact_extraction_error=None,
                ingestion_diagnostics={},
                recommended_next_action=None,
                model_dump=lambda: {"graph_written": True},
            )

    monkeypatch.setattr(document, "_resource_service", lambda: Service())
    monkeypatch.setattr(document, "get_root_runtime", lambda: object())
    monkeypatch.setattr(document, "resolve_pot_id", lambda _host, _pot: "pot-1")

    result = CliRunner().invoke(
        app, ["document", "ingest", "manual.pdf", "--doc", "manual", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    assert seen["vision_provider"] == "local"
    assert seen["extract_facts"] is True
    assert seen["model"] == "gemini-3.5-flash-lite"
    assert seen["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert seen["provider"] == "openai"
    assert seen["key"] == "saved-secret"
    assert seen["vision_model"] == "moondream"
    assert os.environ.get("POTPIE_FACTS_LLM_API_KEY") is None


def test_document_ingest_explicit_opt_out_overrides_saved_defaults(
    tmp_path, monkeypatch
) -> None:
    _isolate_home(tmp_path, monkeypatch)
    config = LocalConfigService()
    config.set("document.vision_provider", "local")
    config.set("document.extract_facts", "on")
    seen = {}

    class Service:
        def ingest_file(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(graph_written=True, errors=[], model_dump=lambda: {})

    monkeypatch.setattr(document, "_resource_service", lambda: Service())
    monkeypatch.setattr(document, "_emit_import_report", lambda _report: None)
    monkeypatch.setattr(document, "get_root_runtime", lambda: object())
    monkeypatch.setattr(document, "resolve_pot_id", lambda _host, _pot: "pot-1")

    result = CliRunner().invoke(
        app,
        [
            "document",
            "ingest",
            "manual.pdf",
            "--doc",
            "manual",
            "--pot",
            "pot-1",
            "--vision-provider",
            "none",
            "--no-extract-facts",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["vision_provider"] == "none"
    assert seen["extract_facts"] is False


def test_document_setup_refuses_json_prompt_without_persisting(
    tmp_path, monkeypatch
) -> None:
    _isolate_home(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["--json", "document", "setup"])

    assert result.exit_code != 0
    assert json.loads(result.output)["code"] == "interactive_required"
    assert not (tmp_path / "home" / "config.json").exists()
