"""P9 use-case readers.

Each reader queries the canonical claim store via
:class:`ClaimQueryPort`, builds :class:`Candidate`s with use-case
specific scope-overlap / similarity signals, and runs them through
the uniform :class:`RankingService` (P7).

Readers are intentionally side-effect free (no Cypher, no HTTP, no
filesystem) so the test surface is the in-memory claim store + a
deterministic ranker.
"""

from potpie_context_engine.application.readers.coding_preferences import (
    CodingPreferencesReader,
)
from potpie_context_engine.application.readers.decisions import DecisionsReader
from potpie_context_engine.application.readers.docs import DocsReader
from potpie_context_engine.application.readers.features import FeaturesReader
from potpie_context_engine.application.readers.infra_topology import InfraTopologyReader
from potpie_context_engine.application.readers.owners import OwnersReader
from potpie_context_engine.application.readers.prior_bugs import PriorBugsReader
from potpie_context_engine.application.readers.timeline_reader import TimelineReader

__all__ = [
    "CodingPreferencesReader",
    "DecisionsReader",
    "DocsReader",
    "FeaturesReader",
    "InfraTopologyReader",
    "OwnersReader",
    "PriorBugsReader",
    "TimelineReader",
]
