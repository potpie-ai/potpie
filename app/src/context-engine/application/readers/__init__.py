"""P9 use-case readers.

Each reader queries the canonical claim store via
:class:`ClaimQueryPort`, builds :class:`Candidate`s with use-case
specific scope-overlap / similarity signals, and runs them through
the uniform :class:`RankingService` (P7).

Readers are intentionally side-effect free (no Cypher, no HTTP, no
filesystem) so the test surface is the in-memory claim store + a
deterministic ranker.
"""

from application.readers.coding_preferences import CodingPreferencesReader
from application.readers.infra_topology import InfraTopologyReader
from application.readers.prior_bugs import PriorBugsReader
from application.readers.timeline_reader import TimelineReader

__all__ = [
    "CodingPreferencesReader",
    "InfraTopologyReader",
    "PriorBugsReader",
    "TimelineReader",
]
