"""Immutable graph definitions and additive extensions.

The legacy module-level catalogs remain compatibility aliases, but runtime
composition uses :class:`GraphDefinition` so two differently extended graph
runtimes can safely coexist in one interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol

from potpie_context_engine.core.graph_contract import ONTOLOGY_VERSION
from potpie_context_engine.core.graph_views import GRAPH_VIEWS, GraphViewSpec
from potpie_context_engine.core.ontology import (
    ACTIVITY_ENDPOINT,
    BASE_GRAPH_LABELS,
    CODE_GRAPH_LABELS,
    EDGE_TYPES,
    ENTITY_TYPES,
    RECORD_TYPES,
    SCOPE_ENDPOINT,
    WILDCARD_ENDPOINT,
    EdgeTypeSpec,
    EntityTypeSpec,
    RecordTypeSpec,
)


class GraphDefinitionError(ValueError):
    """A graph definition is ambiguous or internally incoherent."""


GraphReaderFactory = Callable[..., Any]
_ENTITY_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class GraphReader(Protocol):
    def read(self, request: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class GraphReaderSpec:
    """One named reader registered for an include family.

    ``factory`` may be ``None`` for the OSS declarations. The engine supplies
    their concrete factories at composition time. Extensions normally provide
    a callable accepting ``claim_query`` and optionally ``ranker``.
    """

    name: str
    include: str
    factory: GraphReaderFactory | Any | None = None


def _freeze_entity(spec: EntityTypeSpec) -> EntityTypeSpec:
    return replace(
        spec,
        lifecycle_transitions=MappingProxyType(
            {
                str(source): frozenset(targets)
                for source, targets in spec.lifecycle_transitions.items()
            }
        ),
    )


def _freeze_view(spec: GraphViewSpec) -> GraphViewSpec:
    return replace(spec, extra=MappingProxyType(dict(spec.extra)))


def _named_specs(
    values: Mapping[str, Any] | Iterable[Any] | None, *, attribute: str
) -> dict[str, Any]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return dict(values)
    normalized: dict[str, Any] = {}
    for value in values:
        name = str(getattr(value, attribute))
        if name in normalized:
            raise GraphDefinitionError(f"duplicate {attribute} name {name!r}")
        normalized[name] = value
    return normalized


def _reader_specs(
    readers: (
        Mapping[str, GraphReaderSpec | GraphReaderFactory | Any]
        | Iterable[GraphReaderSpec]
        | None
    ),
) -> Mapping[str, GraphReaderSpec]:
    normalized: dict[str, GraphReaderSpec] = {}
    if readers is not None and not isinstance(readers, Mapping):
        iterable_readers = readers
        reader_mapping: dict[str, GraphReaderSpec] = {}
        for reader in iterable_readers:
            if reader.include in reader_mapping:
                raise GraphDefinitionError(
                    f"duplicate reader include {reader.include!r}"
                )
            reader_mapping[reader.include] = reader
        readers = reader_mapping
    reader_names: set[str] = set()
    for key, value in dict(readers or {}).items():
        if isinstance(value, GraphReaderSpec):
            spec = value
        else:
            spec = GraphReaderSpec(name=str(key), include=str(key), factory=value)
        if key not in {spec.name, spec.include}:
            raise GraphDefinitionError(
                f"reader mapping key {key!r} must match reader name "
                f"{spec.name!r} or include {spec.include!r}"
            )
        if spec.include in normalized:
            raise GraphDefinitionError(f"duplicate reader include {spec.include!r}")
        if spec.name in reader_names:
            raise GraphDefinitionError(f"duplicate reader name {spec.name!r}")
        normalized[spec.include] = spec
        reader_names.add(spec.name)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class GraphExtension:
    """An additive, versioned set of graph contract entries."""

    name: str
    version: str
    entity_types: Mapping[str, EntityTypeSpec] | Iterable[EntityTypeSpec] = field(
        default_factory=dict
    )
    edge_types: Mapping[str, EdgeTypeSpec] | Iterable[EdgeTypeSpec] = field(
        default_factory=dict
    )
    views: Mapping[str, GraphViewSpec] | Iterable[GraphViewSpec] = field(
        default_factory=dict
    )
    record_types: Mapping[str, RecordTypeSpec] | Iterable[RecordTypeSpec] = field(
        default_factory=dict
    )
    readers: (
        Mapping[str, GraphReaderSpec | GraphReaderFactory | Any]
        | Iterable[GraphReaderSpec]
    ) = field(default_factory=dict)
    # Consumer-friendly vocabulary aliases.
    entities: Mapping[str, EntityTypeSpec] | Iterable[EntityTypeSpec] = field(
        default_factory=dict, repr=False
    )
    predicates: Mapping[str, EdgeTypeSpec] | Iterable[EdgeTypeSpec] = field(
        default_factory=dict, repr=False
    )
    records: Mapping[str, RecordTypeSpec] | Iterable[RecordTypeSpec] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise GraphDefinitionError("extension name must be non-empty")
        if not self.version.strip():
            raise GraphDefinitionError(
                f"extension {self.name!r} must declare a non-empty version"
            )
        entity_types = _named_specs(self.entity_types, attribute="label")
        entity_aliases = _named_specs(self.entities, attribute="label")
        _reject_collisions("entity type", entity_types, entity_aliases)
        entity_types.update(entity_aliases)
        edge_types = _named_specs(self.edge_types, attribute="edge_type")
        edge_aliases = _named_specs(self.predicates, attribute="edge_type")
        _reject_collisions("predicate", edge_types, edge_aliases)
        edge_types.update(edge_aliases)
        views = _named_specs(self.views, attribute="name")
        record_types = _named_specs(self.record_types, attribute="record_type")
        record_aliases = _named_specs(self.records, attribute="record_type")
        _reject_collisions("record type", record_types, record_aliases)
        record_types.update(record_aliases)
        frozen_entities = MappingProxyType(
            {name: _freeze_entity(spec) for name, spec in entity_types.items()}
        )
        frozen_edges = MappingProxyType(edge_types)
        frozen_views = MappingProxyType(
            {name: _freeze_view(spec) for name, spec in views.items()}
        )
        frozen_records = MappingProxyType(record_types)
        object.__setattr__(self, "entity_types", frozen_entities)
        object.__setattr__(self, "edge_types", frozen_edges)
        object.__setattr__(self, "views", frozen_views)
        object.__setattr__(self, "record_types", frozen_records)
        object.__setattr__(self, "readers", _reader_specs(self.readers))
        object.__setattr__(self, "entities", frozen_entities)
        object.__setattr__(self, "predicates", frozen_edges)
        object.__setattr__(self, "records", frozen_records)


@dataclass(frozen=True, slots=True)
class GraphDefinition:
    """A complete immutable graph contract plus all derived runtime indexes."""

    ontology_version: str
    entity_types: Mapping[str, EntityTypeSpec]
    edge_types: Mapping[str, EdgeTypeSpec]
    views: Mapping[str, GraphViewSpec]
    record_types: Mapping[str, RecordTypeSpec]
    readers: Mapping[str, GraphReaderSpec] = field(default_factory=dict)
    extensions: Mapping[str, str] = field(default_factory=dict)

    identity_by_label: Mapping[str, Any] = field(init=False, repr=False)
    entity_by_key_prefix: Mapping[str, str] = field(init=False, repr=False)
    singleton_predicates: frozenset[str] = field(init=False)
    view_by_include: Mapping[str, str] = field(init=False, repr=False)
    public_record_types: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        entities = MappingProxyType(
            {name: _freeze_entity(spec) for name, spec in self.entity_types.items()}
        )
        edges = MappingProxyType(dict(self.edge_types))
        views = MappingProxyType(
            {name: _freeze_view(spec) for name, spec in self.views.items()}
        )
        records = MappingProxyType(dict(self.record_types))
        readers = _reader_specs(self.readers)
        extensions = MappingProxyType(dict(self.extensions))
        object.__setattr__(self, "entity_types", entities)
        object.__setattr__(self, "edge_types", edges)
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "record_types", records)
        object.__setattr__(self, "readers", readers)
        object.__setattr__(self, "extensions", extensions)

        errors = _coherence_errors(
            entities=entities,
            edges=edges,
            views=views,
            records=records,
            readers=readers,
        )
        if errors:
            raise GraphDefinitionError(
                "graph definition is incoherent:\n  - " + "\n  - ".join(errors)
            )

        prefixes: dict[str, str] = {}
        identities: dict[str, Any] = {}
        for label, spec in entities.items():
            previous = prefixes.get(spec.key_prefix)
            if previous is not None:
                raise GraphDefinitionError(
                    f"entity key prefix {spec.key_prefix!r} is shared by "
                    f"{previous!r} and {label!r}"
                )
            prefixes[spec.key_prefix] = label
            identities[label] = MappingProxyType(
                {
                    "label": label,
                    "identity_class": spec.identity_class,
                    "key_prefix": spec.key_prefix,
                    "authoritative_source": spec.authoritative_source,
                }
            )
        object.__setattr__(self, "entity_by_key_prefix", MappingProxyType(prefixes))
        object.__setattr__(self, "identity_by_label", MappingProxyType(identities))
        object.__setattr__(
            self,
            "singleton_predicates",
            frozenset(name for name, spec in edges.items() if spec.singleton),
        )
        object.__setattr__(
            self,
            "view_by_include",
            MappingProxyType({spec.v1_include: name for name, spec in views.items()}),
        )
        object.__setattr__(
            self,
            "public_record_types",
            frozenset(name for name, spec in records.items() if spec.public),
        )

    def extend(
        self, *extensions: GraphExtension | Iterable[GraphExtension]
    ) -> GraphDefinition:
        """Return a new definition after collision and coherence checks.

        The source definition is never mutated. All collisions are checked
        before the new definition is returned and therefore before runtime
        composition can touch a backend.
        """

        flattened: list[GraphExtension] = []
        for candidate in extensions:
            if isinstance(candidate, GraphExtension):
                flattened.append(candidate)
            else:
                flattened.extend(candidate)

        entities = dict(self.entity_types)
        edges = dict(self.edge_types)
        views = dict(self.views)
        records = dict(self.record_types)
        readers = dict(self.readers)
        versions = dict(self.extensions)
        reader_names = {reader.name for reader in readers.values()}

        for extension in flattened:
            if extension.name in versions:
                raise GraphDefinitionError(
                    f"duplicate extension name {extension.name!r}"
                )
            _reject_collisions("entity type", entities, extension.entity_types)
            _reject_collisions("predicate", edges, extension.edge_types)
            _reject_collisions("view", views, extension.views)
            _reject_collisions("record type", records, extension.record_types)
            _reject_collisions("reader include", readers, extension.readers)
            for reader in extension.readers.values():
                if reader.name in reader_names:
                    raise GraphDefinitionError(f"duplicate reader name {reader.name!r}")
                reader_names.add(reader.name)
            entities.update(extension.entity_types)
            edges.update(extension.edge_types)
            views.update(extension.views)
            records.update(extension.record_types)
            readers.update(extension.readers)
            versions[extension.name] = extension.version

        return GraphDefinition(
            ontology_version=self.ontology_version,
            entity_types=entities,
            edge_types=edges,
            views=views,
            record_types=records,
            readers=readers,
            extensions=versions,
        )

    @property
    def entities(self) -> Mapping[str, EntityTypeSpec]:
        return self.entity_types

    @property
    def predicates(self) -> Mapping[str, EdgeTypeSpec]:
        return self.edge_types

    @property
    def records(self) -> Mapping[str, RecordTypeSpec]:
        return self.record_types

    def status_metadata(self) -> dict[str, Any]:
        return {
            "ontology_version": self.ontology_version,
            "extensions": dict(self.extensions),
            "entity_type_count": len(self.entity_types),
            "predicate_count": len(self.edge_types),
            "view_count": len(self.views),
            "record_type_count": len(self.record_types),
            "reader_count": len(self.readers),
        }

    def validate_structural_mutations(
        self,
        entity_upserts: Iterable[Any],
        edge_upserts: Iterable[Any],
        edge_deletes: Iterable[Any] = (),
    ) -> list[str]:
        """Validate a lowered batch against this definition only."""

        errors: list[str] = []
        labels_by_key: dict[str, tuple[str, ...]] = {}
        allowed_noncanonical = {"Entity", *CODE_GRAPH_LABELS}
        for entity in entity_upserts:
            labels = tuple(str(label) for label in entity.labels)
            labels_by_key[entity.entity_key] = labels
            unknown = [
                label
                for label in labels
                if label not in self.entity_types and label not in allowed_noncanonical
            ]
            if unknown:
                errors.append(
                    f"{entity.entity_key}: unknown entity labels "
                    f"{', '.join(sorted(unknown))}"
                )
            for label in labels:
                spec = self.entity_types.get(label)
                if spec is None:
                    continue
                missing = sorted(
                    key
                    for key in spec.required_properties
                    if entity.properties.get(key) in (None, "")
                )
                if missing:
                    errors.append(
                        f"{entity.entity_key}: {label} missing required properties "
                        f"{', '.join(missing)}"
                    )
        for edge in (*tuple(edge_upserts), *tuple(edge_deletes)):
            spec = self.edge_types.get(edge.edge_type)
            if spec is None:
                errors.append(f"unknown edge type {edge.edge_type!r}")
                continue
            properties = getattr(edge, "properties", {})
            missing = sorted(
                key
                for key in spec.required_properties
                if properties.get(key) in (None, "")
            )
            if missing:
                errors.append(
                    f"{edge.edge_type}: missing required properties "
                    f"{', '.join(missing)}"
                )
            from_labels = labels_by_key.get(edge.from_entity_key)
            to_labels = labels_by_key.get(edge.to_entity_key)
            if (
                from_labels is not None
                and to_labels is not None
                and not spec.allows(from_labels, to_labels)
            ):
                errors.append(
                    f"{edge.edge_type} does not allow "
                    f"{'/'.join(from_labels)} -> {'/'.join(to_labels)}"
                )
        return errors


def _reject_collisions(
    kind: str, current: Mapping[str, Any], additions: Mapping[str, Any]
) -> None:
    duplicates = sorted(set(current) & set(additions))
    if duplicates:
        raise GraphDefinitionError(
            f"duplicate {kind} name(s): {', '.join(repr(name) for name in duplicates)}"
        )


def _coherence_errors(
    *,
    entities: Mapping[str, EntityTypeSpec],
    edges: Mapping[str, EdgeTypeSpec],
    views: Mapping[str, GraphViewSpec],
    records: Mapping[str, RecordTypeSpec],
    readers: Mapping[str, GraphReaderSpec],
) -> list[str]:
    errors: list[str] = []
    for name, spec in entities.items():
        if name != spec.label:
            errors.append(f"entity key {name!r} does not match label {spec.label!r}")
        if not _ENTITY_LABEL_RE.fullmatch(spec.label):
            errors.append(f"entity label {spec.label!r} is not a safe identifier")
    special_endpoints = {
        WILDCARD_ENDPOINT,
        SCOPE_ENDPOINT,
        ACTIVITY_ENDPOINT,
        *BASE_GRAPH_LABELS,
        *CODE_GRAPH_LABELS,
    }
    for name, spec in edges.items():
        if name != spec.edge_type:
            errors.append(
                f"predicate key {name!r} does not match edge_type {spec.edge_type!r}"
            )
        if not _PREDICATE_RE.fullmatch(spec.edge_type):
            errors.append(f"predicate {spec.edge_type!r} is not a safe identifier")
        for left, right in spec.allowed_pairs:
            for endpoint in (left, right):
                if endpoint not in entities and endpoint not in special_endpoints:
                    errors.append(
                        f"predicate {name!r} references unknown endpoint {endpoint!r}"
                    )
    include_to_view: dict[str, str] = {}
    for name, spec in views.items():
        if name != spec.name:
            errors.append(f"view key {name!r} does not match name {spec.name!r}")
        previous = include_to_view.get(spec.v1_include)
        if previous is not None:
            errors.append(
                f"views {previous!r} and {name!r} share include {spec.v1_include!r}"
            )
        include_to_view[spec.v1_include] = name
        if spec.backed and spec.v1_include not in readers:
            errors.append(
                f"backed view {name!r} has no reader for include {spec.v1_include!r}"
            )
    for name, spec in records.items():
        if name != spec.record_type:
            errors.append(
                f"record key {name!r} does not match record_type {spec.record_type!r}"
            )
        if spec.anchor_label not in entities:
            errors.append(
                f"record {name!r} references unknown anchor {spec.anchor_label!r}"
            )
        if spec.emits_predicate and spec.emits_predicate not in edges:
            errors.append(
                f"record {name!r} emits unknown predicate {spec.emits_predicate!r}"
            )
        if spec.reader_include and spec.reader_include not in readers:
            errors.append(
                f"record {name!r} references unknown reader include "
                f"{spec.reader_include!r}"
            )
    return errors


_DEFAULT_READER_INCLUDES = frozenset(
    spec.v1_include for spec in GRAPH_VIEWS.values() if spec.backed
)

DEFAULT_GRAPH_DEFINITION = GraphDefinition(
    ontology_version=ONTOLOGY_VERSION,
    entity_types=ENTITY_TYPES,
    edge_types=EDGE_TYPES,
    views=GRAPH_VIEWS,
    record_types=RECORD_TYPES,
    readers={
        include: GraphReaderSpec(name=include, include=include)
        for include in _DEFAULT_READER_INCLUDES
    },
)


__all__ = [
    "DEFAULT_GRAPH_DEFINITION",
    "GraphDefinition",
    "GraphDefinitionError",
    "GraphExtension",
    "GraphReader",
    "GraphReaderFactory",
    "GraphReaderSpec",
]
