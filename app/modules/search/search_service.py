import os
from typing import Dict, List

from sqlalchemy import delete, or_
from sqlalchemy.orm import Session

from app.modules.search.search_models import SearchIndex


class SearchService:
    def __init__(self, db: Session):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db

    async def commit_indices(self):
        self.db.commit()

    async def search_codebase(self, project_id: str, query: str) -> List[Dict]:
        # Split the query into words
        query_words = query.lower().split()

        # Perform the search
        results = (
            self.db.query(SearchIndex)
            .filter(
                SearchIndex.project_id == project_id,
                or_(
                    *[
                        or_(
                            SearchIndex.name.ilike(f"%{word}%"),
                            SearchIndex.file_path.ilike(f"%{word}%"),
                            SearchIndex.content.ilike(f"%{word}%"),
                        )
                        for word in query_words
                    ]
                ),
            )
            .all()
        )

        # Format and sort the results
        formatted_results = []
        ids = []
        for result in results:
            relevance = self._calculate_relevance(result, query_words)
            if result.node_id not in ids:
                formatted_results.append(
                    {
                        "node_id": result.node_id,
                        "name": result.name,
                        "file_path": (
                            result.file_path.split(self.project_path, 1)[-1].split(
                                "/", 2
                            )[
                                -1
                            ]  # ensure that your project path value does not end with a /
                            if self.project_path in result.file_path
                            else result.file_path
                        ),
                        "content": result.content,
                        "match_type": self._determine_match_type(result, query_words),
                        "relevance": relevance,
                    }
                )
            ids.append(result.node_id)

        # Sort results by relevance
        formatted_results.sort(key=lambda x: x["relevance"], reverse=True)

        return formatted_results[:10]

    def _calculate_relevance(
        self, result: SearchIndex, query_words: List[str]
    ) -> float:
        relevance = 0

        for word in query_words:
            if word in result.name.lower():
                relevance += 3  # Highest relevance for name match
            if word in result.file_path.lower():
                relevance += 2  # Medium relevance for file path match
            if word in result.content.lower():
                relevance += 1  # Lowest relevance for content match

        # Adjust relevance based on how many query words match
        relevance *= len(
            [word for word in query_words if word in result.content.lower()]
        ) / len(query_words)

        # Adjust relevance based on how close the match is to the full string
        name_similarity = self._string_similarity(" ".join(query_words), result.name)
        file_path_similarity = self._string_similarity(
            " ".join(query_words), result.file_path
        )

        relevance += (name_similarity + file_path_similarity) / 2

        return relevance

    def _determine_match_type(self, result: SearchIndex, query_words: List[str]) -> str:
        if all(word in result.content.lower() for word in query_words):
            return "Exact Match"
        return "Partial Match"

    def _string_similarity(self, a: str, b: str) -> float:
        # Simple string similarity calculation
        a = a.lower()
        b = b.lower()
        return len(set(a) & set(b)) / float(len(set(a) | set(b)))

    def delete_project_index(self, project_id: str):
        # Delete all search index entries for the given project_id
        delete_stmt = delete(SearchIndex).where(SearchIndex.project_id == project_id)
        self.db.execute(delete_stmt)
        self.db.commit()

    async def bulk_create_search_indices(self, nodes: List[Dict]):
        # Create index entries for all nodes in bulk
        self.db.bulk_insert_mappings(SearchIndex, nodes)

    async def clone_search_indices(self, input_project_id: str, output_project_id: str):
        """Clone all search indices from input project to output project."""
        # Query all search indices for the input project
        source_indices = (
            self.db.query(SearchIndex)
            .filter(SearchIndex.project_id == input_project_id)
            .all()
        )

        # Prepare bulk insert data
        cloned_indices = [
            {
                "project_id": output_project_id,
                "node_id": index.node_id,
                "name": index.name,
                "file_path": index.file_path,
                "content": index.content,
            }
            for index in source_indices
        ]

        # Bulk insert the cloned indices if there are any
        if cloned_indices:
            self.db.bulk_insert_mappings(SearchIndex, cloned_indices)
            await self.commit_indices()
