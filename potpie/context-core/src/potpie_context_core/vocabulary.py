"""Canonical spellings for enumerated graph filters.

``graph search-entities --type repository`` used to return zero rows at exit 0:
the service compares the requested label with hydrated labels case-sensitively,
and ``repository`` matches nothing while ``Repository`` matches two. The same
held for predicates (``policy_applies_to`` vs ``POLICY_APPLIES_TO``) and for a
misspelling (``Repositry``), which looked exactly like a valid filter with no
matches. A confident empty answer is the worst of the three outcomes: the
caller cannot tell "nothing matched" from "the filter was impossible".

These helpers resolve an *exact* variant — case, surrounding whitespace, and
for predicates the space/hyphen spellings ``normalize_edge_name`` already
accepts — onto the canonical token, and report a short list of candidates for
anything else. They never pick a near miss: a candidate list is guidance, the
choice stays with the caller. Entity IDs, environments, source systems and
free-text queries are literal values and are not routed through here.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Final, Iterable

from potpie_context_core.graph_views import GRAPH_VIEWS
from potpie_context_core.ontology import EDGE_TYPES, ENTITY_TYPES, normalize_edge_name
from potpie_context_core.semantic_mutation_lowering import claim_subgraph_names

#: How many alternatives an unknown-value refusal lists. Enough to correct a
#: typo, few enough that the error stays smaller than the catalog it replaces.
CANDIDATE_LIMIT: Final[int] = 6


@dataclass(frozen=True, slots=True)
class VocabularyMatch:
    """Outcome of resolving one filter value against a closed vocabulary."""

    requested: str
    canonical: str | None
    """The canonical token, or ``None`` when the value is unknown."""

    candidates: tuple[str, ...] = ()
    """Bounded alternatives for an unknown value (empty when resolved)."""

    @property
    def changed(self) -> bool:
        return self.canonical is not None and self.canonical != self.requested


def local_entity_types() -> tuple[str, ...]:
    return tuple(label for label, spec in ENTITY_TYPES.items() if spec.public)


def local_predicates() -> tuple[str, ...]:
    return tuple(name for name, spec in EDGE_TYPES.items() if spec.public)


def local_claim_subgraphs() -> tuple[str, ...]:
    """Every value a stored claim's ``subgraph`` property can take.

    The union of the read-view subgraphs and the slices the mutation lowering
    assigns (``memory``, ``admin``, ``code_topology`` …), because a claim that
    ``search-entities --subgraph`` can filter on was written by one of the two.
    """
    names = {spec.subgraph for spec in GRAPH_VIEWS.values()}
    names.update(claim_subgraph_names())
    return tuple(sorted(names))


def close_candidates(
    value: str, known: Iterable[str], *, limit: int = CANDIDATE_LIMIT
) -> tuple[str, ...]:
    """A bounded, deterministic set of plausible alternatives for ``value``.

    Close matches first (so a typo sees its correction), then any known token
    that shares the value as a substring, then the first few of the vocabulary
    in sorted order so the list is never empty for a caller who had no idea.
    """
    pool = sorted({str(item) for item in known if str(item)})
    if not pool:
        return ()
    needle = value.strip().lower()
    out: list[str] = []
    by_lower = {item.lower(): item for item in pool}
    for match in difflib.get_close_matches(needle, list(by_lower), n=limit, cutoff=0.6):
        out.append(by_lower[match])
    for item in pool:
        if len(out) >= limit:
            break
        if item in out:
            continue
        lowered = item.lower()
        if needle and (needle in lowered or lowered in needle):
            out.append(item)
    for item in pool:
        if len(out) >= limit:
            break
        if item not in out:
            out.append(item)
    return tuple(out[:limit])


def _exact_case_insensitive(value: str, known: Iterable[str]) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    pool = [str(item) for item in known]
    if cleaned in pool:
        return cleaned
    lowered = cleaned.lower()
    matches = [item for item in pool if item.lower() == lowered]
    # Two canonical tokens differing only by case would make the alias
    # ambiguous; the ontology has none, and refusing is the right answer if
    # one ever appears.
    return matches[0] if len(matches) == 1 else None


def resolve_entity_type(
    value: str, *, known: Iterable[str] | None = None
) -> VocabularyMatch:
    pool = tuple(known) if known is not None else local_entity_types()
    canonical = _exact_case_insensitive(value, pool)
    if canonical is not None:
        return VocabularyMatch(requested=value, canonical=canonical)
    return VocabularyMatch(
        requested=value, canonical=None, candidates=close_candidates(value, pool)
    )


def resolve_predicate(
    value: str, *, known: Iterable[str] | None = None
) -> VocabularyMatch:
    pool = tuple(known) if known is not None else local_predicates()
    cleaned = value.strip()
    if cleaned in pool:
        return VocabularyMatch(requested=value, canonical=cleaned)
    normalized = normalize_edge_name(cleaned) if cleaned else ""
    if normalized in pool:
        return VocabularyMatch(requested=value, canonical=normalized)
    return VocabularyMatch(
        requested=value,
        canonical=None,
        candidates=close_candidates(normalized or cleaned, pool),
    )


def resolve_claim_subgraph(
    value: str, *, known: Iterable[str] | None = None
) -> VocabularyMatch:
    pool = tuple(known) if known is not None else local_claim_subgraphs()
    cleaned = value.strip().replace("-", "_").replace(" ", "_")
    canonical = _exact_case_insensitive(cleaned, pool)
    if canonical is not None:
        return VocabularyMatch(requested=value, canonical=canonical)
    return VocabularyMatch(
        requested=value, canonical=None, candidates=close_candidates(cleaned, pool)
    )


def vocabulary_from_catalog(payload: Any) -> dict[str, tuple[str, ...]]:
    """The advertised vocabulary carried by a ``graph catalog`` result.

    Read off the serving host so a client with an older in-process ontology
    still accepts a valid server extension. Missing sections come back empty
    and the caller falls back to its local registry for them.
    """
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if not isinstance(payload, dict):
        return {}
    entity_types = tuple(
        str(item.get("label"))
        for item in payload.get("entity_types", ())
        if isinstance(item, dict) and item.get("label")
    )
    predicates = tuple(
        str(item.get("name"))
        for item in payload.get("predicates", ())
        if isinstance(item, dict) and item.get("name")
    )
    subgraphs = tuple(
        sorted(
            {
                str(item.get("subgraph"))
                for item in payload.get("views", ())
                if isinstance(item, dict) and item.get("subgraph")
            }
            | {
                str(item.get("category"))
                for item in payload.get("predicates", ())
                if isinstance(item, dict) and item.get("category")
            }
        )
    )
    return {
        "entity_types": entity_types,
        "predicates": predicates,
        "subgraphs": subgraphs,
    }


__all__ = [
    "CANDIDATE_LIMIT",
    "VocabularyMatch",
    "close_candidates",
    "local_claim_subgraphs",
    "local_entity_types",
    "local_predicates",
    "resolve_claim_subgraph",
    "resolve_entity_type",
    "resolve_predicate",
    "vocabulary_from_catalog",
]
