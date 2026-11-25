"""Local vision setup checks the model Ollama can actually serve."""

from __future__ import annotations

import httpx
import pytest

from potpie_context_engine.adapters.outbound.resources.enrichers import (
    vision_httpx_ollama,
)

pytestmark = pytest.mark.unit


def test_ollama_model_available_matches_installed_model_tags(monkeypatch) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "moondream:latest"}]})

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_: client_type(transport=httpx.MockTransport(respond)),
    )

    assert vision_httpx_ollama.ollama_model_available("moondream") is True
    assert vision_httpx_ollama.ollama_model_available("moondream:latest") is True
    assert vision_httpx_ollama.ollama_model_available("moondream:older") is False
    assert vision_httpx_ollama.ollama_model_available("llava") is False


def test_local_vision_rejects_coordinate_only_output(tmp_path, monkeypatch) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"test image")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, json={"response": "[0.39, 0.13, 0.6, 0.26]"})

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_: client_type(transport=httpx.MockTransport(respond)),
    )
    assert vision_httpx_ollama.caption_image_local(image) == ""


def test_ollama_vision_capability_is_checked(monkeypatch) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={"capabilities": ["completion", "vision"]})

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_: client_type(transport=httpx.MockTransport(respond)),
    )
    assert vision_httpx_ollama.ollama_model_supports_vision("moondream")
