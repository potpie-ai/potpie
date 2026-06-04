"""Precision sub-axis — purity against the expected negative class.

Precision = 1 - (distractor / signal hits). Reported as 0..100 alongside
the primary axis score. Distractor events are the negative class (see
bench-plan §5.3); without them every scenario would only grade recall on
small graphs and recall-without-precision regressions would be
invisible.

Precision attaches to two axes:

- **Ingestion**: did the graph stay clean of entities / labels that
  ``graph_must_not_contain`` declared?
- **Retrieval**: did the response avoid citing event ids listed under
  ``must_not_cite_event_id`` (and source_refs that map to no expected
  event)?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionOutcome:
    relevant: int
    distractors: int

    @property
    def score(self) -> float:
        total = self.relevant + self.distractors
        if total <= 0:
            return 100.0
        return round(100.0 * self.relevant / total, 1)


def precision_score(relevant: int, distractors: int) -> float:
    return PrecisionOutcome(relevant=relevant, distractors=distractors).score
