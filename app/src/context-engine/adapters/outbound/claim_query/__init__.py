"""Claim-query adapters.

The in-memory fake lives here so production code never imports from
``tests/``; both readers and tests share the same implementation.
"""

from adapters.outbound.claim_query.in_memory import InMemoryClaimQueryStore

__all__ = ["InMemoryClaimQueryStore"]
