from __future__ import annotations

import pytest

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinitionError,
    GraphExtension,
    GraphReaderSpec,
)
from potpie_context_engine.core.graph_views import GraphViewSpec
from potpie_context_engine.core.identity import IdentityClass
from potpie_context_engine.core.ontology import EdgeTypeSpec, EntityTypeSpec


def _extension(name: str = "widgets") -> GraphExtension:
    return GraphExtension(
        name=name,
        version="1.0",
        entity_types={
            "Widget": EntityTypeSpec(
                label="Widget",
                category="topology",
                description="A test widget.",
                identity_class=IdentityClass.SLUG_ALIAS,
                key_prefix="widget",
                identity_policy="widget:<slug>",
            )
        },
        edge_types={
            "CONNECTS_WIDGET": EdgeTypeSpec(
                edge_type="CONNECTS_WIDGET",
                description="Connects two widgets.",
                allowed_pairs=(("Widget", "Widget"),),
            )
        },
        views={
            "widgets.network": GraphViewSpec(
                name="widgets.network",
                subgraph="widgets",
                view="network",
                v1_include="widgets",
                description="Widget network.",
                backed=True,
            )
        },
        readers={"widgets": lambda claim_query: claim_query},
    )


def test_extension_returns_new_immutable_definition() -> None:
    extended = DEFAULT_GRAPH_DEFINITION.extend(_extension())

    assert "Widget" not in DEFAULT_GRAPH_DEFINITION.entity_types
    assert "Widget" in extended.entity_types
    assert extended.entity_by_key_prefix["widget"] == "Widget"
    assert extended.extensions == {"widgets": "1.0"}
    with pytest.raises(TypeError):
        extended.entity_types["Other"] = extended.entity_types["Widget"]  # type: ignore[index]


@pytest.mark.parametrize(
    "extension",
    [
        GraphExtension(
            name="duplicate-entity",
            version="1",
            entity_types={"Service": DEFAULT_GRAPH_DEFINITION.entity_types["Service"]},
        ),
        GraphExtension(
            name="duplicate-view",
            version="1",
            views={
                "features.feature_context": DEFAULT_GRAPH_DEFINITION.views[
                    "features.feature_context"
                ]
            },
        ),
    ],
)
def test_catalog_collisions_fail_during_extend(extension: GraphExtension) -> None:
    with pytest.raises(GraphDefinitionError, match="duplicate"):
        DEFAULT_GRAPH_DEFINITION.extend(extension)


def test_duplicate_extension_name_is_rejected() -> None:
    once = DEFAULT_GRAPH_DEFINITION.extend(_extension())
    with pytest.raises(GraphDefinitionError, match="duplicate extension"):
        once.extend(_extension())


def test_duplicate_names_in_iterable_extension_are_rejected() -> None:
    first = EntityTypeSpec(
        label="Duplicate",
        category="topology",
        description="First.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="duplicate",
        identity_policy="duplicate:<slug>",
    )
    second = EntityTypeSpec(
        label="Duplicate",
        category="topology",
        description="Second.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="duplicate-alt",
        identity_policy="duplicate-alt:<slug>",
    )

    with pytest.raises(GraphDefinitionError, match="duplicate label"):
        GraphExtension(
            name="duplicate-iterable",
            version="1",
            entity_types=(first, second),
        )


def test_duplicate_reader_includes_in_iterable_extension_are_rejected() -> None:
    with pytest.raises(GraphDefinitionError, match="duplicate reader include"):
        GraphExtension(
            name="duplicate-readers",
            version="1",
            readers=(
                GraphReaderSpec(name="first", include="widgets"),
                GraphReaderSpec(name="second", include="widgets"),
            ),
        )
