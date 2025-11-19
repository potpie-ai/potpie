from typing import List, Tuple
import logging

from app.modules.parsing.graph_construction.directory_scanner_service import (
    DirectoryScannerService, DirectoryWorkUnit
)
from app.modules.parsing.graph_construction.neo4j_state_service import Neo4jStateService
from app.modules.parsing.parsing_work_unit_model import ParsingWorkUnit
from app.modules.parsing.parsing_session_model import ParsingSession
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class WorkUnitBootstrapService:
    """
    Service for bootstrapping resumable parsing state from Neo4j.

    Used when resuming parsing that was started before work unit tracking
    was implemented.
    """

    def __init__(
        self,
        db: Session,
        neo4j_state_service: Neo4jStateService,
        repo_path: str
    ):
        self.db = db
        self.neo4j_state_service = neo4j_state_service
        self.repo_path = repo_path

    def bootstrap_from_neo4j(
        self,
        project_id: str,
        commit_id: str,
        user_id: str
    ) -> Tuple[ParsingSession, List[DirectoryWorkUnit]]:
        """
        Bootstrap work unit state from Neo4j and file system.

        NEW APPROACH: Creates work units without querying Neo4j upfront.
        Individual workers will check their specific files when they run.

        Process:
        1. Scan repository to create work units (like normal parsing)
        2. Create session and work unit records (status=pending for all)
        3. Return all work units for processing
        4. Workers will check Neo4j for their specific files

        Returns:
            (ParsingSession, List of DirectoryWorkUnit objects with DB IDs)
        """
        logger.info(f"Bootstrapping work units from Neo4j for project {project_id}")

        # Step 1: Scan repository and create work units
        scanner = DirectoryScannerService(self.repo_path)
        work_units = scanner.scan_and_divide()

        logger.info(f"Scanned repository: {len(work_units)} work units, {scanner.total_files} files")

        # Step 2: Create session record
        session = ParsingSession(
            project_id=project_id,
            commit_id=commit_id,
            session_number=1,
            coordinator_task_id="bootstrap",
            total_work_units=len(work_units),
            total_files=scanner.total_files,
            stage='parsing',
            processed_files=0  # Will be updated as workers complete
        )
        self.db.add(session)
        self.db.flush()

        # Step 3: Create work unit records (all pending, workers will check Neo4j)
        incomplete_units = []

        for i, work_unit in enumerate(work_units):
            db_work_unit = ParsingWorkUnit(
                project_id=project_id,
                commit_id=commit_id,
                work_unit_index=i,
                directory_path=work_unit.path,
                files=work_unit.files,
                file_count=work_unit.file_count,
                depth=work_unit.depth,
                status='pending',  # All pending - workers will check
                attempt_count=0
            )

            self.db.add(db_work_unit)
            self.db.flush()  # Get ID

            # Create DirectoryWorkUnit with DB ID attached
            incomplete_unit = DirectoryWorkUnit(
                path=work_unit.path,
                files=work_unit.files,
                file_count=work_unit.file_count,
                depth=work_unit.depth
            )
            incomplete_unit.id = db_work_unit.id
            incomplete_units.append(incomplete_unit)

        # Commit all work units
        self.db.commit()

        logger.info(
            f"Bootstrap complete: {len(work_units)} work units created, "
            f"all marked pending (workers will check Neo4j)"
        )

        return session, incomplete_units

    def detect_parsing_stage(self, project_id: str) -> str:
        """
        Determine what stage of parsing a project is in based on Neo4j state.

        Returns:
            'scanning', 'parsing', 'resolving', 'inferring', or 'finalizing'
        """
        stats = self.neo4j_state_service.get_parsing_statistics(project_id)

        if stats['total_nodes'] == 0:
            return 'scanning'

        parsing_pct = stats['parsing_percentage']
        inference_pct = stats['inference_percentage']

        if parsing_pct < 100:
            return 'parsing'
        elif inference_pct == 0:
            return 'resolving'  # Parsing done, inference not started
        elif inference_pct < 100:
            return 'inferring'
        else:
            return 'finalizing'
