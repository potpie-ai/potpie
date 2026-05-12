"""Neo4j structural read port (adapter-internal).

Phase 1 of the rebuild made Graphiti the only writer; the structural
adapter is restricted to reads. The mutation methods (``upsert_entities``,
``upsert_edges``, ``delete_edges``, ``apply_invalidations``,
``reset_pot``) moved into the Graphiti adapter
(``adapters/outbound/graphiti/canonical_writer.py`` plus
``EpisodicGraphPort.apply_*``).

Application-level code depends only on
:class:`domain.ports.context_graph.ContextGraphPort`. This port is
consumed by the Graphiti context-graph adapter and the intelligence
adapter for exact structural reads. Phase 3 of the rebuild plan will
collapse these read methods into a per-family ``ContextReader`` registry
and retire this port.
"""

from typing import Any, Protocol


class StructuralReadPort(Protocol):
    def get_change_history(
        self,
        pot_id: str,
        function_name: str | None,
        file_path: str | None,
        limit: int,
        repo_name: str | None = None,
        pr_number: int | None = None,
        as_of: str | None = None,
        node_uuids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_timeline(
        self,
        pot_id: str,
        *,
        since_iso: str,
        until_iso: str,
        limit: int,
        user: str | None = None,
        feature: str | None = None,
        file_path: str | None = None,
        branch: str | None = None,
        verbs: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def get_file_owners(
        self,
        pot_id: str,
        file_path: str,
        limit: int,
        repo_name: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_decisions(
        self,
        pot_id: str,
        file_path: str | None,
        function_name: str | None,
        limit: int,
        repo_name: str | None = None,
        pr_number: int | None = None,
        node_uuids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_pr_review_context(
        self,
        pot_id: str,
        pr_number: int,
        repo_name: str | None = None,
    ) -> dict[str, Any]: ...

    def get_pr_diff(
        self,
        pot_id: str,
        pr_number: int,
        file_path: str | None,
        limit: int,
        repo_name: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_project_graph(
        self,
        pot_id: str,
        pr_number: int | None,
        limit: int,
        scope: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def get_debugging_memory(
        self,
        pot_id: str,
        limit: int,
        scope: dict[str, Any] | None = None,
        include: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]: ...

    def get_graph_overview(
        self,
        pot_id: str,
        *,
        top_entities_limit: int = 20,
    ) -> dict[str, Any]: ...

    def expand_causal_neighbours(
        self,
        pot_id: str,
        node_uuids: list[str],
        *,
        depth: int = 1,
    ) -> list[dict[str, Any]]: ...

    def walk_causal_chain_backward(
        self,
        pot_id: str,
        focal_node_uuid: str,
        *,
        max_depth: int = 6,
        as_of_iso: str | None = None,
        window_days: int = 180,
    ) -> list[dict[str, Any]]: ...

    def resolve_entity_uuid_for_service_hint(
        self,
        pot_id: str,
        service_hint: str,
    ) -> str | None: ...

    def get_episodic_entity_node(
        self,
        pot_id: str,
        entity_uuid: str,
    ) -> dict[str, Any] | None: ...
