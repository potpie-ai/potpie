"""Base resource class for PotpieRuntime library."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)


class BaseResource:
    """Base class for all resource classes.

    Provides common functionality and access to core managers.
    Resources are user-agnostic - user context is passed per-operation.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        db_manager: DatabaseManager,
        neo4j_manager: Neo4jManager,
    ):
        """Initialize base resource.

        Args:
            config: Runtime configuration
            db_manager: Database manager for PostgreSQL
            neo4j_manager: Neo4j manager for knowledge graph
        """
        self._config = config
        self._db_manager = db_manager
        self._neo4j_manager = neo4j_manager
