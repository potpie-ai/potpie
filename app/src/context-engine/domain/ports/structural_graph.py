"""Neo4j code graph + Entity bridges (port)."""

from typing import Any, Protocol

from domain.ingestion import BridgeResult


class StructuralGraphPort(Protocol):
    def write_bridges(
        self,
        project_id: str,
        pr_entity_key: str,
        pr_number: int,
        repo_name: str,
        files_with_patches: list[dict[str, Any]],
        review_threads: list[dict[str, Any]],
        merged_at: str | None,
        is_live: bool,
    ) -> BridgeResult:
        ...

    def stamp_pr_entities(
        self,
        project_id: str,
        episode_uuid: str,
        repo_name: str,
        pr_number: int,
        commits: list[dict[str, Any]],
        review_threads: list[dict[str, Any]],
        pr_data: dict[str, Any] | None = None,
        author: str | None = None,
        pr_title: str | None = None,
        issue_comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        ...

    def get_change_history(
        self,
        project_id: str,
        function_name: str | None,
        file_path: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    def get_file_owners(
        self,
        project_id: str,
        file_path: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    def get_decisions(
        self,
        project_id: str,
        file_path: str | None,
        function_name: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    def get_pr_review_context(self, project_id: str, pr_number: int) -> dict[str, Any]:
        ...

    def get_pr_diff(
        self,
        project_id: str,
        pr_number: int,
        file_path: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        ...
