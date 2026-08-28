"""Detect installed document-ingestion optional dependencies."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def documents_capabilities() -> dict[str, bool]:
    return {
        "pypdf": _has_module("pypdf"),
        "pypdfium2": _has_module("pypdfium2"),
        "docling": _has_module("docling"),
        "rapidocr": _has_module("rapidocr_onnxruntime"),
        "httpx": _has_module("httpx"),
    }


def documents_extra_label() -> str:
    caps = documents_capabilities()
    if caps["docling"]:
        return "documents"
    return "base"


def docling_cache_dir() -> Path | None:
    for key in ("DOCLING_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        raw = os.getenv(key)
        if raw:
            return Path(raw).expanduser()
    home = Path.home()
    return home / ".cache" / "docling"


def assess_document_ingest_readiness() -> dict[str, Any]:
    """Dedicated readiness gate for `potpie document ingest`."""
    caps = documents_capabilities()
    host_mode = (os.getenv("CONTEXT_ENGINE_HOST_MODE") or "daemon").strip().lower()
    recommendations: list[str] = []

    checks = {
        "markdown_text": True,
        "html_stdlib": True,
        "documents_extra_docling": caps["docling"],
        "rapidocr": caps["rapidocr"],
        "pdf_degraded_fallback": caps["pypdf"] or caps["pypdfium2"],
        "vision_httpx": caps["httpx"],
        "host_in_process": host_mode == "in_process",
    }

    if not caps["docling"]:
        recommendations.append(
            "install potpie[documents] for PDF provenance, office formats, Docling HTML, and images"
        )
    if not caps["rapidocr"] and caps["docling"]:
        recommendations.append("rapidocr_onnxruntime missing — image OCR may be limited")
    if not caps["httpx"]:
        recommendations.append("httpx missing — vision captions unavailable")
    if host_mode != "in_process":
        recommendations.append(
            "set CONTEXT_ENGINE_HOST_MODE=in_process for document CLI until daemon document RPC ships"
        )

    minimal_ready = checks["markdown_text"] and checks["html_stdlib"]
    full_ready = (
        caps["docling"]
        and caps["rapidocr"]
        and caps["httpx"]
        and checks["host_in_process"]
    )

    return {
        "document_ingest_ready": minimal_ready,
        "full_documents_ready": full_ready,
        "host_mode": host_mode,
        "host_document_cli_ready": checks["host_in_process"],
        "checks": checks,
        "recommendations": recommendations,
    }


def collect_documents_doctor_report() -> dict[str, Any]:
    caps = documents_capabilities()
    label = documents_extra_label()
    cache = docling_cache_dir()
    ingest_readiness = assess_document_ingest_readiness()
    ollama_ok = False
    if caps["httpx"]:
        from potpie_context_engine.adapters.outbound.resources.enrichers.vision_httpx_ollama import (
            ollama_reachable,
        )

        ollama_ok = ollama_reachable()

    recommendations: list[str] = list(ingest_readiness.get("recommendations") or [])
    if not caps["docling"]:
        if "install potpie[documents]" not in " ".join(recommendations):
            recommendations.append(
                "install potpie[documents] for layout PDFs, provenance, OCR, and images (Docling + RapidOCR)"
            )
    if caps["docling"] and cache and not cache.exists():
        recommendations.append(
            f"Docling model cache not found at {cache}; first PDF parse will download models"
        )

    return {
        "tier": label,
        "capabilities": caps,
        "document_ingest": ingest_readiness,
        "docling_cache_dir": str(cache) if cache else None,
        "docling_cache_exists": bool(cache and cache.exists()),
        "ollama_reachable": ollama_ok,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "recommendations": recommendations,
    }


__all__ = [
    "assess_document_ingest_readiness",
    "collect_documents_doctor_report",
    "documents_capabilities",
    "documents_extra_label",
    "docling_cache_dir",
]
