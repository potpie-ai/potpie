"""Coverage sub-axis — recall against the expected positive class.

Coverage = (positives found) / (positives expected). Reported as 0..100
alongside the primary axis score. The plan calls for separate
coverage/precision evaluator files (bench-plan §6.3); this module owns
the shared math, and the primary-axis evaluators in
``ingestion_quality.py`` / ``retrieval.py`` call into it.

Coverage attaches to two axes:

- **Ingestion**: did the post-ingest graph contain the entity / edge
  assertions the scenario declared?
- **Retrieval**: of the event ids the scenario said *must be cited*, how
  many were actually cited?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageOutcome:
    expected: int
    found: int

    @property
    def score(self) -> float:
        if self.expected <= 0:
            return 100.0
        return round(100.0 * min(self.found, self.expected) / self.expected, 1)


def coverage_score(expected: int, found: int) -> float:
    return CoverageOutcome(expected=expected, found=found).score
