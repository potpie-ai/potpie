"""Tests for metadata-aware PDF retrieval evaluation."""

from __future__ import annotations

import pytest

from potpie_context_engine.benchmarks.pdf_retrieval_eval import (
    PdfRetrievalCase,
    evaluate_pdf_case,
)

pytestmark = pytest.mark.unit


def test_pdf_eval_counts_section_title_and_evidence_metadata() -> None:
    case = PdfRetrievalCase(
        case_id="R07",
        category="exact",
        accepted_aliases=("Language Models are Unsupervised Multitask Learners",),
        physical_pages=(1,),
    )
    hits = [
        {
            "rank": 1,
            "payload": {
                "properties": {
                    "section_title": "Language Models are Unsupervised Multitask Learners",
                    "evidence_text": "Language Models are Unsupervised Multitask Learners",
                    "page_numbers": [1],
                }
            },
            "opened": [{"payload": {"content": "Alec Radford et al."}}],
        }
    ]

    result = evaluate_pdf_case(case, hits)

    assert result.alias_rank == 1
    assert result.page_rank == 1
    assert result.citation_accurate is True


def test_pdf_eval_separates_negative_abstention_and_figure_readiness() -> None:
    negative = evaluate_pdf_case(
        PdfRetrievalCase(
            case_id="R28",
            category="negative",
            accepted_aliases=(),
            physical_pages=(),
            unanswerable=True,
        ),
        [],
    )
    figure = evaluate_pdf_case(
        PdfRetrievalCase(
            case_id="R22",
            category="figure",
            accepted_aliases=("WebText train",),
            physical_pages=(9,),
        ),
        [
            {
                "rank": 1,
                "payload": {
                    "properties": {
                        "evidence_text": "Figure 4. WebText train remains below WebText test.",
                        "page_numbers": [9],
                    }
                },
                "opened": [],
            }
        ],
    )

    assert negative.negative_abstained is True
    assert figure.figure_caption_ready is True
    assert figure.alias_rank == 1
