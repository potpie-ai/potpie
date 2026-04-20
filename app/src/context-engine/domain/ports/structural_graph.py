"""Neo4j code graph + Entity bridges (port)."""

from typing import Any, Protocol

from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from domain.ingestion import BridgeResult


class StructuralGraphPort(Protocol):
    def write_bridges(
        self,
        pot_id: str,
        pr_entity_key: str,
        pr_number: int,
        repo_name: str,
        files_with_patches: list[dict[str, Any]],
        review_threads: list[dict[str, Any]],
        merged_at: str | None,
        is_live: bool,
    ) -> BridgeResult: ...

    def stamp_pr_entities(
        self,
        pot_id: str,
        episode_uuid: str,
        repo_name: str,
        pr_number: int,
        commits: list[dict[str, Any]],
        review_threads: list[dict[str, Any]],
        pr_data: dict[str, Any] | None = None,
        author: str | None = None,
        pr_title: str | None = None,
        issue_comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]: ...

    def get_change_history(
        self,
        pot_id: str,
        function_name: str | None,
        file_path: str | None,
        limit: int,
        repo_name: str | None = None,
        pr_number: int | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]: ...

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

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        """Remove structural graph nodes (``Entity`` / ``FILE`` / ``NODE``) scoped to this pot."""
        ...

    def upsert_entities(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int: ...

    def upsert_edges(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int: ...

    def delete_edges(
        self,
        pot_id: str,
        items: list[EdgeDelete],
        provenance: ProvenanceRef,
    ) -> int: ...

    def apply_invalidations(
        self,
        pot_id: str,
        items: list[InvalidationOp],
        provenance: ProvenanceRef,
    ) -> int: ...
