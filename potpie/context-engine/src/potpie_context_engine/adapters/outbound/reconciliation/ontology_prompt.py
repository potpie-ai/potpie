"""Canonical modeling vocabulary for the ingestion agent, derived from the catalog."""

from potpie_context_core.ontology import EDGE_TYPES, ENTITY_TYPES


def render_ingestion_ontology() -> str:
    lines = [
        "## Canonical ingestion ontology",
        (
            "Select the most specific source-supported type and predicate. "
            "Examples and playbooks do not limit this vocabulary."
        ),
        "Entity | exact key prefix | description",
    ]
    for label, spec in ENTITY_TYPES.items():
        if spec.public:
            lines.append(f"{label} | {spec.key_prefix}: | {spec.description}")
    lines.append("Predicate | allowed subject -> object | description")
    for name, spec in EDGE_TYPES.items():
        if spec.public:
            pairs = "; ".join(f"{a} -> {b}" for a, b in spec.allowed_pairs)
            lines.append(f"{name} | {pairs} | {spec.description}")
    scopes = ", ".join(label for label, spec in ENTITY_TYPES.items() if spec.scope)
    activities = ", ".join(
        label for label, spec in ENTITY_TYPES.items() if spec.is_activity
    )
    lines.extend(
        [
            f"@Scope = {scopes}; @Activity = {activities}; * = any endpoint.",
            (
                "Current behavior and dependencies are facts; an explicit choice "
                "with rationale is a Decision; only explicit reusable prescriptions "
                "are Preference/Policy claims. A capability needs Feature relations, "
                "not a generic note. A source can support several separate claims. "
                "Resolve identities before linking; defer unsupported classifications "
                "instead of forcing them into preferences or generic associations."
            ),
        ]
    )
    return "\n".join(lines)
