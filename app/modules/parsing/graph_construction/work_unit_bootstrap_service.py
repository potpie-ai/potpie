from typing import List, Tuple
import logging

from app.modules.parsing.graph_construction.directory_scanner_service import (
    DirectoryScannerService, DirectoryWorkUnit
)
from app.modules.parsing.graph_construction.neo4j_state_service import Neo4jStateService
from app.modules.parsing.parsing_work_unit_model import ParsingWorkUnit
from app.modules.parsing.parsing_session_model import ParsingSession
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

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
        1. Check for existing incomplete session - if found, reuse it
        2. Check for completed sessions - if found, increment session_number
        3. Scan repository to create work units (like normal parsing)
        4. Create session and work unit records (status=pending for all)
           - Handles race conditions via IntegrityError catch-and-retry
        5. Return all work units for processing
        6. Workers will check Neo4j for their specific files

        Race condition handling:
        - If two bootstrap calls run concurrently, one will succeed and one will
          hit IntegrityError due to uq_project_commit_session constraint
        - The failing call will rollback, re-query for the now-committed session,
          and reuse it instead of crashing

        Returns:
            (ParsingSession, List of DirectoryWorkUnit objects with DB IDs)
        """
        logger.info(f"Bootstrapping work units from Neo4j for project {project_id}")

        # Step 1: Check for existing incomplete session first
        # IMPORTANT: Use NULL-safe comparison for commit_id
        session_query = self.db.query(ParsingSession).filter(
            ParsingSession.project_id == project_id,
            ParsingSession.completed_at.is_(None)
        )
        if commit_id is not None:
            session_query = session_query.filter(ParsingSession.commit_id == commit_id)
        else:
            session_query = session_query.filter(ParsingSession.commit_id.is_(None))
        existing_incomplete_session = session_query.order_by(
            ParsingSession.session_number.desc()
        ).first()

        if existing_incomplete_session:
            logger.info(
                f"Found existing incomplete session {existing_incomplete_session.id} "
                f"(session_number={existing_incomplete_session.session_number}), reusing it"
            )
            # Reuse existing session and query existing work units
            session = existing_incomplete_session

            # Query existing incomplete work units (NULL-safe comparison for commit_id)
            work_units_query = self.db.query(ParsingWorkUnit).filter(
                ParsingWorkUnit.project_id == project_id,
                ParsingWorkUnit.status.in_(['pending', 'failed'])
            )
            if commit_id is not None:
                work_units_query = work_units_query.filter(ParsingWorkUnit.commit_id == commit_id)
            else:
                work_units_query = work_units_query.filter(ParsingWorkUnit.commit_id.is_(None))
            db_work_units = work_units_query.all()

            # Convert to DirectoryWorkUnit objects
            incomplete_units = []
            for db_work_unit in db_work_units:
                incomplete_unit = DirectoryWorkUnit(
                    path=db_work_unit.directory_path,
                    files=db_work_unit.files,
                    file_count=db_work_unit.file_count,
                    depth=db_work_unit.depth
                )
                incomplete_unit.id = db_work_unit.id
                incomplete_units.append(incomplete_unit)

            logger.info(
                f"Reusing session: {len(incomplete_units)} incomplete work units found"
            )
            return session, incomplete_units

        # Step 2: No incomplete session found, check for completed sessions to get next session_number
        # IMPORTANT: Use NULL-safe comparison for commit_id
        max_session_query = self.db.query(
            func.max(ParsingSession.session_number)
        ).filter(
            ParsingSession.project_id == project_id
        )
        if commit_id is not None:
            max_session_query = max_session_query.filter(ParsingSession.commit_id == commit_id)
        else:
            max_session_query = max_session_query.filter(ParsingSession.commit_id.is_(None))
        max_session_number = max_session_query.scalar()

        next_session_number = (max_session_number or 0) + 1

        if max_session_number:
            logger.info(
                f"Found completed sessions (max session_number={max_session_number}), "
                f"creating new session with session_number={next_session_number}"
            )
        else:
            logger.info(f"No existing sessions found, creating first session")

        # Step 3: Scan repository and create work units
        scanner = DirectoryScannerService(self.repo_path)
        work_units = scanner.scan_and_divide()

        logger.info(f"Scanned repository: {len(work_units)} work units, {scanner.total_files} files")

        # Step 4: Create new session record with incremented session_number
        # Wrap in try/except to handle race condition where another transaction
        # creates the same session_number before we commit
        try:
            session = ParsingSession(
                project_id=project_id,
                commit_id=commit_id,
                session_number=next_session_number,
                coordinator_task_id="bootstrap",
                total_work_units=len(work_units),
                total_files=scanner.total_files,
                stage='parsing',
                processed_files=0  # Will be updated as workers complete
            )
            self.db.add(session)
            self.db.flush()
        except IntegrityError as e:
            # Another transaction created a session with this session_number
            # Roll back and re-query for the now-committed session
            logger.warning(
                f"IntegrityError creating session (race condition detected): {e}. "
                f"Rolling back and re-querying for existing session"
            )
            self.db.rollback()

            # Re-query for the session that the other transaction created (NULL-safe)
            race_session_query = self.db.query(ParsingSession).filter(
                ParsingSession.project_id == project_id,
                ParsingSession.completed_at.is_(None)
            )
            if commit_id is not None:
                race_session_query = race_session_query.filter(ParsingSession.commit_id == commit_id)
            else:
                race_session_query = race_session_query.filter(ParsingSession.commit_id.is_(None))
            existing_session = race_session_query.order_by(
                ParsingSession.session_number.desc()
            ).first()

            if existing_session:
                logger.info(
                    f"Found session created by concurrent transaction: "
                    f"session_id={existing_session.id}, session_number={existing_session.session_number}"
                )
                session = existing_session

                # Query existing incomplete work units (NULL-safe comparison for commit_id)
                race_work_units_query = self.db.query(ParsingWorkUnit).filter(
                    ParsingWorkUnit.project_id == project_id,
                    ParsingWorkUnit.status.in_(['pending', 'failed'])
                )
                if commit_id is not None:
                    race_work_units_query = race_work_units_query.filter(
                        ParsingWorkUnit.commit_id == commit_id
                    )
                else:
                    race_work_units_query = race_work_units_query.filter(
                        ParsingWorkUnit.commit_id.is_(None)
                    )
                db_work_units = race_work_units_query.all()

                # Convert to DirectoryWorkUnit objects
                incomplete_units = []
                for db_work_unit in db_work_units:
                    incomplete_unit = DirectoryWorkUnit(
                        path=db_work_unit.directory_path,
                        files=db_work_unit.files,
                        file_count=db_work_unit.file_count,
                        depth=db_work_unit.depth
                    )
                    incomplete_unit.id = db_work_unit.id
                    incomplete_units.append(incomplete_unit)

                logger.info(
                    f"Reusing session after race condition: {len(incomplete_units)} incomplete work units found"
                )
                return session, incomplete_units
            else:
                # This should never happen, but re-raise if we can't find the session
                logger.error("Failed to find session after IntegrityError - this should not happen")
                raise

        # Step 5: Create work unit records (all pending, workers will check Neo4j)
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
