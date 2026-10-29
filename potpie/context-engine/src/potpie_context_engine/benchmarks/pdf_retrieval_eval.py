"""Deterministic, metadata-aware evaluation for PDF retrieval stress runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class PdfRetrievalCase:
    case_id: str
    category: str
    accepted_aliases: tuple[str, ...]
    physical_pages: tuple[int, ...]
    unanswerable: bool = False


@dataclass(frozen=True, slots=True)
class PdfCaseResult:
    alias_rank: int | None
    page_rank: int | None
    citation_accurate: bool
    negative_abstained: bool
    figure_caption_ready: bool


def evaluate_pdf_case(
    case: PdfRetrievalCase, hits: Iterable[dict[str, Any]]
) -> PdfCaseResult:
    """Evaluate a case using both raw chunks and additive search metadata."""
    hits = list(hits)
    alias_rank: int | None = None
    page_rank: int | None = None
    figure_caption_ready = False
    for hit in hits:
        rank = int(hit.get("rank") or 0)
        if rank <= 0:
            continue
        payload = hit.get("payload") or {}
        properties = payload.get("properties") if isinstance(payload, dict) else {}
        properties = properties if isinstance(properties, dict) else {}
        opened_payloads = [
            opened.get("payload")
            for opened in hit.get("opened") or []
            if isinstance(opened, dict) and isinstance(opened.get("payload"), dict)
        ]
        sources = [payload, properties, *opened_payloads]
        haystack = " ".join(
            str(value)
            for source in sources
            for value in source.values()
            if value not in (None, "", [], {})
        ).casefold()
        pages = _pages(sources)
        if case.accepted_aliases and any(
            alias.casefold() in haystack for alias in case.accepted_aliases
        ):
            alias_rank = min(alias_rank or rank, rank)
        if case.physical_pages and set(case.physical_pages) & pages:
            page_rank = min(page_rank or rank, rank)
        if case.category == "figure" and _has_figure_evidence(sources):
            figure_caption_ready = True

    return PdfCaseResult(
        alias_rank=alias_rank,
        page_rank=page_rank,
        citation_accurate=(
            bool(page_rank) if case.physical_pages else True
        ),
        negative_abstained=case.unanswerable and alias_rank is None and not list(hits),
        figure_caption_ready=figure_caption_ready,
    )


def _pages(sources: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for source in sources:
        for key in ("page_numbers", "pages"):
            values = source.get(key)
            if isinstance(values, list):
                pages.update(int(value) for value in values if isinstance(value, int))
        citation = source.get("citation")
        if isinstance(citation, dict) and isinstance(citation.get("pages"), list):
            pages.update(
                int(value)
                for value in citation["pages"]
                if isinstance(value, int)
            )
    return pages


def _has_figure_evidence(sources: list[dict[str, Any]]) -> bool:
    text = " ".join(
        str(source.get(key) or "")
        for source in sources
        for key in ("evidence_text", "visual_caption", "caption", "content")
    ).casefold()
    return "figure" in text or "visual caption" in text or "caption" in text


__all__ = [
    "PdfCaseResult",
    "PdfRetrievalCase",
    "evaluate_pdf_case",
]
