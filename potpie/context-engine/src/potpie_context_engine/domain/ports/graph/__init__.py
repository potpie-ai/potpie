"""The ``GraphBackend`` capability ports.

One swappable backend = six narrow capability ports. ``mutation`` +
``claim_query`` are the canonical source of truth; ``semantic`` ``inspection``
``analytics`` ``snapshot`` are rebuildable projections. See ``backend.py``.

Import directly from the capability modules (``potpie_context_engine.domain.ports.graph.backend``,
``.mutation``, ...); the canonical ``ClaimQueryPort`` lives at
``potpie_context_engine.domain.ports.claim_query``.
"""
