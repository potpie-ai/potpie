"""Re-export of the canonical :class:`ClaimQueryPort` under the GraphBackend
capability namespace.

``ClaimQueryPort`` is one of the six ``GraphBackend`` capabilities (and, with
``GraphMutationPort``, the source of truth). It already lives at
``domain.ports.claim_query`` with 100+ importers, so it is re-exported here
rather than moved — new ``GraphBackend`` code can import it from the capability
package while existing imports keep working.
"""

from __future__ import annotations

from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow

__all__ = ["ClaimQueryFilter", "ClaimQueryPort", "ClaimRow"]
