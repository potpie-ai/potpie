"""Optional engine factory; the base definition and default recipes stay intact."""

from potpie_context_core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
    GraphExtension,
    GraphReaderSpec,
)
from potpie_context_core.protocols import (
    PROTOCOL_EDGE_TYPES,
    PROTOCOL_ENTITY_TYPES,
    PROTOCOL_VIEW,
    PROTOCOLS_VERSION,
)

from potpie_context_engine.application.readers.protocols import ProtocolsReader


def protocols_definition(
    base: GraphDefinition = DEFAULT_GRAPH_DEFINITION,
) -> GraphDefinition:
    return base.extend(
        GraphExtension(
            name="protocols",
            version=PROTOCOLS_VERSION,
            entity_types=PROTOCOL_ENTITY_TYPES,
            edge_types=PROTOCOL_EDGE_TYPES,
            views=(PROTOCOL_VIEW,),
            readers=(
                GraphReaderSpec(
                    name="protocols.message_context",
                    include="protocols",
                    factory=ProtocolsReader,
                ),
            ),
        )
    )
