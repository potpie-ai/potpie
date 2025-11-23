import logging
import os
from typing import Any, Dict

from sqlalchemy import func
from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_parsing",
    ignore_result=False,  # Store results for monitoring and debugging
)
def process_parsing(
    self,
    repo_details: Dict[str, Any],
    user_id: str,
    user_email: str,
    project_id: str,
    cleanup_graph: bool = True,
) -> None:
    logger.info(f"Task received: Starting parsing process for project {project_id}")
    try:
        parsing_service = ParsingService(self.db, user_id)

        async def run_parsing():
            import time

            start_time = time.time()

            await parsing_service.parse_directory(
                ParsingRequest(**repo_details),
                user_id,
                user_email,
                project_id,
                cleanup_graph,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(
                f"Parsing process took {elapsed_time:.2f} seconds for project {project_id}"
            )

        # Use BaseTask's long-lived event loop for consistency
        self.run_async(run_parsing())
        logger.info(f"Parsing process completed for project {project_id}")
    except Exception as e:
        logger.error(f"Error during parsing for project {project_id}: {str(e)}")
        raise


def resume_parsing_session(db_session, session, task_instance) -> Dict[str, Any]:
    """
    Resume parsing from an existing session.

    Handles both:
    - Sessions created by new system (work units in DB)
    - Sessions bootstrapped from Neo4j (work units reconstructed)
    """

    logger.info(
        f"Resuming session {session.id} (session #{session.session_number}) "
        f"at stage: {session.stage}"
    )

    if session.stage == 'parsing':
        return resume_parsing_stage(db_session, session, task_instance)
    elif session.stage == 'aggregating':
        # For now, just log - aggregation resume not yet implemented
        logger.warning("Aggregation stage resume not yet implemented")
        return {'success': False, 'error': 'Aggregation resume not implemented'}
    else:
        logger.warning(f"Unknown or unimplemented stage: {session.stage}")
        return {'success': False, 'error': f'Stage {session.stage} resume not implemented'}


def resume_parsing_stage(db_session, session, task_instance) -> Dict[str, Any]:
    """
    Resume parsing stage by spawning tasks for incomplete work units.

    Now properly clones the repository by fetching project metadata from database.
    """
    from app.modules.parsing.parsing_work_unit_model import ParsingWorkUnit
    from app.modules.projects.projects_model import Project
    from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
    from app.modules.parsing.graph_construction.parsing_service import ParsingRequest
    from datetime import datetime
    from celery import chord

    # Step 1: Query incomplete work units
    incomplete_units = db_session.query(ParsingWorkUnit).filter(
        ParsingWorkUnit.project_id == session.project_id,
        ParsingWorkUnit.commit_id == session.commit_id,
        ParsingWorkUnit.status.in_(['pending', 'failed'])
    ).all()

    if not incomplete_units:
        logger.info("No incomplete work units, all work completed")
        session.completed_at = datetime.utcnow()
        session.stage = 'completed'
        db_session.commit()
        return {
            'success': True,
            'session_id': str(session.id),
            'status': 'completed',
            'message': 'All work units completed'
        }

    logger.info(f"Found {len(incomplete_units)} incomplete work units to resume")

    # Step 2: Fetch project metadata to reconstruct repo_details
    project = db_session.query(Project).filter(
        Project.id == session.project_id
    ).first()

    if not project:
        error_msg = f"Project {session.project_id} not found in database"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'session_id': str(session.id)
        }

    logger.info(f"Found project: {project.repo_name} (branch: {project.branch_name})")

    # Step 3: Reconstruct repo_details from project
    repo_details = {
        'repo_name': project.repo_name,
        'branch_name': project.branch_name,
        'commit_id': session.commit_id,
        'repo_path': project.repo_path,
    }

    # Step 4: Clone/setup repository
    parse_helper = ParseHelper(db_session)

    async def setup_repo():
        return await parse_helper.clone_or_copy_repository(
            ParsingRequest(**repo_details),
            user_id=project.user_id
        )

    async def setup_project_dir(repo, auth):
        return await parse_helper.setup_project_directory(
            repo,
            project.branch_name,
            auth,
            ParsingRequest(**repo_details),
            user_id=project.user_id,
            project_id=session.project_id,
            commit_id=session.commit_id
        )

    try:
        logger.info(f"Cloning repository for resume: {project.repo_name}")
        repo, owner, auth = task_instance.run_async(setup_repo())
        repo_path, _ = task_instance.run_async(setup_project_dir(repo, auth))
        logger.info(f"Repository cloned successfully to: {repo_path}")
    except Exception as e:
        logger.exception(f"Failed to clone repository for resume: {e}")
        return {
            'success': False,
            'error': f"Repository clone failed: {str(e)}",
            'session_id': str(session.id)
        }

    # Step 5: Reset Redis counter for resume (important for coordination)
    from app.celery.coordination import ParsingCoordinator
    try:
        redis_client = task_instance.app.backend.client
        ParsingCoordinator.reset_counter(redis_client, session.project_id)
        logger.info(f"Reset Redis completion counter for project {session.project_id}")
    except Exception as redis_error:
        logger.error(f"Failed to reset Redis counter: {redis_error}")

    # Step 6: Reset work units to pending and increment attempt counter
    for work_unit in incomplete_units:
        work_unit.status = 'pending'
        work_unit.attempt_count += 1

    db_session.commit()

    # Step 7: Create task signatures using existing parse_directory_unit
    parsing_tasks = []
    for work_unit in incomplete_units:
        task = parse_directory_unit.s(
            work_unit_index=work_unit.work_unit_index,
            directory_path=work_unit.directory_path,
            files=work_unit.files,
            repo_path=repo_path,  # Now properly cloned!
            project_id=session.project_id,
            user_id=project.user_id,
            repo_name=project.repo_name,
            commit_id=session.commit_id,
            work_unit_db_id=str(work_unit.id)
        )
        parsing_tasks.append(task)

    # Step 8: Execute as group (no callback - workers coordinate via Redis)
    from celery import group
    group_task = group(parsing_tasks).apply_async()

    logger.info(
        f"Dispatched {len(incomplete_units)} work units for resume (group_id={group_task.id})"
    )

    # Update session
    session.coordinator_task_id = str(group_task.id)
    session.updated_at = datetime.utcnow()
    db_session.commit()

    return {
        'success': True,
        'session_id': str(session.id),
        'group_task_id': str(group_task.id),
        'resumed_work_units': len(incomplete_units),
        'status': 'resuming'
    }


def bootstrap_and_resume(
    task_instance,
    neo4j_state_service,
    repo_details: dict,
    user_id: str,
    user_email: str,
    project_id: str,
    commit_id: str,
    cleanup_graph: bool
) -> Dict[str, Any]:
    """
    Bootstrap work unit state from Neo4j and resume parsing.

    Called when:
    - No session exists in database
    - But Neo4j has partial parsing data
    - User wants to resume
    """
    from app.modules.parsing.graph_construction.work_unit_bootstrap_service import (
        WorkUnitBootstrapService
    )
    from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
    from celery import chord

    logger.info(f"Starting bootstrap process for project {project_id}")

    # Setup repository (same as normal parsing)
    parse_helper = ParseHelper(task_instance.db)

    async def setup_repo():
        return await parse_helper.clone_or_copy_repository(
            ParsingRequest(**repo_details),
            user_id
        )

    async def setup_project_dir(repo, auth):
        return await parse_helper.setup_project_directory(
            repo,
            repo_details.get('branch_name'),
            auth,
            ParsingRequest(**repo_details),
            user_id,
            project_id,
            commit_id=commit_id
        )

    repo, owner, auth = task_instance.run_async(setup_repo())
    project_path, actual_commit_id = task_instance.run_async(setup_project_dir(repo, auth))

    logger.info(f"Repository setup complete: {project_path}")

    # Use the actual commit ID from setup if the passed one was None
    if not commit_id:
        commit_id = actual_commit_id
        logger.info(f"Using commit ID from repository setup: {commit_id}")

    # Optional: Selective cleanup of failed work units
    if cleanup_graph:
        logger.warning("Cleanup requested but skipping to preserve partial state")
        # Could implement selective cleanup here if needed

    # Bootstrap work units from Neo4j + filesystem
    bootstrap_service = WorkUnitBootstrapService(
        db=task_instance.db,
        neo4j_state_service=neo4j_state_service,
        repo_path=project_path
    )

    session, incomplete_units = bootstrap_service.bootstrap_from_neo4j(
        project_id=project_id,
        commit_id=commit_id,
        user_id=user_id
    )

    logger.info(
        f"Bootstrap complete: session {session.id}, "
        f"{len(incomplete_units)} work units to process"
    )

    if not incomplete_units:
        logger.info("All work units already completed, moving to aggregation")
        # TODO: Trigger aggregation stage
        return {
            'success': True,
            'session_id': str(session.id),
            'status': 'aggregating',
            'message': 'All work units completed, starting aggregation'
        }

    # Reset Redis counter for bootstrap (important for coordination)
    from app.celery.coordination import ParsingCoordinator
    try:
        redis_client = task_instance.app.backend.client
        ParsingCoordinator.reset_counter(redis_client, project_id)
        logger.info(f"Reset Redis completion counter for project {project_id}")
    except Exception as redis_error:
        logger.error(f"Failed to reset Redis counter: {redis_error}")

    # Create group with incomplete work units (no callback - workers coordinate via Redis)
    parsing_tasks = []
    for i, work_unit in enumerate(incomplete_units):
        # Use database ID from the DirectoryWorkUnit object
        task = parse_directory_unit.s(
            work_unit_index=i,  # Ordinal index for logging
            directory_path=work_unit.path,
            files=work_unit.files,
            repo_path=project_path,
            project_id=project_id,
            user_id=user_id,
            repo_name=repo_details.get('repo_name', ''),
            commit_id=commit_id,  # Actual commit ID
            work_unit_db_id=str(work_unit.id)  # UUID for DB operations
        )
        parsing_tasks.append(task)

    # Execute as group (no callback - workers coordinate via Redis)
    from celery import group
    group_task = group(parsing_tasks).apply_async()

    logger.info(
        f"Dispatched {len(incomplete_units)} work units for bootstrap (group_id={group_task.id})"
    )

    # Update session with group ID
    session.coordinator_task_id = str(group_task.id)
    task_instance.db.commit()

    return {
        'success': True,
        'session_id': str(session.id),
        'group_task_id': str(group_task.id),
        'bootstrapped': True,
        'incomplete_work_units': len(incomplete_units),
        'status': 'resuming'
    }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_parsing_distributed",
    time_limit=7200,  # 2 hours
    ignore_result=False,  # Store results for monitoring and chord coordination
)
def process_parsing_distributed(
    self,
    repo_details: dict,
    user_id: str,
    user_email: str,
    project_id: str,
    cleanup_graph: bool = False,
    resume: bool = True  # Enable resume by default
) -> Dict[str, Any]:
    """
    Master coordinator task for distributed repository parsing.

    This task:
    1. Scans repository structure
    2. Divides into work units
    3. Spawns worker tasks for each unit
    4. Uses Celery chord to trigger callback after workers complete

    Args:
        repo_details: Repository information
        user_id: User ID
        user_email: User email
        project_id: Project ID
        cleanup_graph: Whether to cleanup existing graph

    Returns:
        Dictionary with setup results (actual parsing happens asynchronously)
    """
    from app.modules.parsing.graph_construction.directory_scanner_service import (
        DirectoryScannerService
    )
    from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
    from app.core.config_provider import config_provider
    from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
    from celery import chord
    import time

    logger.info(
        f"Starting distributed parsing for project {project_id}"
    )
    start_time = time.time()

    try:
        commit_id = repo_details.get('commit_id')

        # NEW: Check for existing incomplete session in database
        from app.modules.parsing.parsing_session_model import ParsingSession

        existing_session = self.db.query(ParsingSession).filter(
            ParsingSession.project_id == project_id,
            ParsingSession.commit_id == commit_id,
            ParsingSession.completed_at.is_(None)
        ).order_by(ParsingSession.session_number.desc()).first()

        if resume and existing_session:
            logger.info(f"Resuming existing session {existing_session.id}")
            return resume_parsing_session(self.db, existing_session, self)

        # NEW: Check for partial parsing in Neo4j (bootstrap case)
        # Only bootstrap if:
        # 1. Resume is enabled
        # 2. No existing session found
        # 3. cleanup_graph is False (if cleanup requested, do fresh parse)
        # 4. Neo4j has nodes (potential incomplete parse)
        if resume and not existing_session and not cleanup_graph:
            from app.modules.parsing.graph_construction.neo4j_state_service import Neo4jStateService

            neo4j_config = config_provider.get_neo4j_config()
            neo4j_state_service = Neo4jStateService(
                neo4j_uri=neo4j_config["uri"],
                neo4j_user=neo4j_config["username"],
                neo4j_password=neo4j_config["password"]
            )

            try:
                has_partial_state = neo4j_state_service.has_any_nodes(project_id)

                if has_partial_state:
                    logger.info(
                        f"Detected partial parsing in Neo4j for project {project_id}. "
                        f"Bootstrapping work unit state..."
                    )
                    return bootstrap_and_resume(
                        self,
                        neo4j_state_service,
                        repo_details,
                        user_id,
                        user_email,
                        project_id,
                        commit_id,
                        cleanup_graph
                    )
                else:
                    logger.info(
                        f"No partial state found in Neo4j for project {project_id}, "
                        f"proceeding with fresh parse"
                    )
            finally:
                neo4j_state_service.close()
        elif resume and not existing_session and cleanup_graph:
            logger.info(
                f"cleanup_graph=True for project {project_id}, "
                f"skipping resume/bootstrap and proceeding with fresh parse"
            )

        # Continue with normal fresh parsing...
        # Step 1: Setup repository
        parse_helper = ParseHelper(self.db)

        async def setup_repo():
            return await parse_helper.clone_or_copy_repository(
                ParsingRequest(**repo_details),
                user_id
            )

        async def setup_project_dir(repo, auth):
            return await parse_helper.setup_project_directory(
                repo,
                repo_details.get('branch_name'),
                auth,
                ParsingRequest(**repo_details),
                user_id,
                project_id,
                commit_id=repo_details.get('commit_id')
            )

        # Clone repository
        repo, owner, auth = self.run_async(setup_repo())
        logger.info("Repository cloned/copied successfully")

        # Setup project directory and get actual commit ID
        project_path, actual_commit_id = self.run_async(setup_project_dir(repo, auth))
        logger.info(f"Project directory setup: {project_path}")

        # Use the actual commit ID from repository setup if not provided
        if not commit_id:
            commit_id = actual_commit_id
            logger.info(f"Using commit ID from repository setup: {commit_id}")

        # Cleanup existing graph if requested
        if cleanup_graph:
            logger.info("Cleaning up existing graph")
            neo4j_config = config_provider.get_neo4j_config()
            code_graph_service = CodeGraphService(
                neo4j_config["uri"],
                neo4j_config["username"],
                neo4j_config["password"],
                self.db
            )
            code_graph_service.cleanup_graph(project_id)

        # Step 2: Scan and divide repository
        scanner = DirectoryScannerService(project_path)
        work_units = scanner.scan_and_divide()

        logger.info(
            f"Created {len(work_units)} work units for {scanner.total_files} files"
        )

        # Step 3: Create task group for parallel processing
        # NOTE: We no longer use chord with callback to avoid OOM from loading
        # hundreds of task results. Instead, workers coordinate via Redis counter
        # and the last worker triggers finalization.
        parsing_tasks = []
        for i, work_unit in enumerate(work_units):
            task = parse_directory_unit.s(
                work_unit_index=i,
                directory_path=work_unit.path,
                files=work_unit.files,
                repo_path=project_path,
                project_id=project_id,
                user_id=user_id,
                repo_name=repo_details.get('repo_name', ''),
                commit_id=commit_id  # Pass commit_id for worker coordination
            )
            parsing_tasks.append(task)

        # Step 4: Execute as group (no callback - workers coordinate via Redis)
        from celery import group
        group_task = group(parsing_tasks).apply_async()

        logger.info(
            f"Dispatched {len(work_units)} work units as group (group_id={group_task.id})"
        )

        # Create or update session for worker coordination
        from app.modules.parsing.parsing_session_model import ParsingSession
        from datetime import datetime

        # Check if session already exists (from bootstrap/resume)
        session = self.db.query(ParsingSession).filter(
            ParsingSession.project_id == project_id,
            ParsingSession.commit_id == commit_id,
            ParsingSession.completed_at.is_(None)
        ).first()

        if not session:
            # Create new session
            session_number = self.db.query(
                func.max(ParsingSession.session_number)
            ).filter(
                ParsingSession.project_id == project_id,
                ParsingSession.commit_id == commit_id
            ).scalar() or 0

            session = ParsingSession(
                project_id=project_id,
                commit_id=commit_id,
                session_number=session_number + 1,
                coordinator_task_id=str(group_task.id),
                total_work_units=len(work_units),
                total_files=scanner.total_files,
                stage='parsing',
                processed_files=0
            )
            self.db.add(session)
        else:
            # Update existing session
            session.coordinator_task_id = str(group_task.id)
            session.total_work_units = len(work_units)
            session.total_files = scanner.total_files

        self.db.commit()

        elapsed = time.time() - start_time

        logger.info(
            f"Distributed parsing setup complete: {len(work_units)} work units dispatched in {elapsed:.2f}s"
        )

        return {
            'success': True,
            'project_id': project_id,
            'work_units': len(work_units),
            'total_files': scanner.total_files,
            'setup_duration_seconds': elapsed,
            'group_task_id': group_task.id,
            'session_id': str(session.id),
            'status': 'processing'
        }

    except Exception as e:
        logger.exception("Error in distributed parsing coordinator")

        # Reset Redis counter so retry can work
        from app.celery.coordination import ParsingCoordinator
        try:
            redis_client = self.app.backend.client
            ParsingCoordinator.reset_counter(redis_client, project_id)
        except Exception as redis_error:
            logger.error(f"Failed to reset Redis counter: {redis_error}")

        return {
            'success': False,
            'error': str(e),
            'project_id': project_id
        }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.parse_directory_unit",
    time_limit=1800,  # 30 minutes
    ignore_result=False,  # Results must be stored for chord callback aggregation
)
def parse_directory_unit(
    self,
    work_unit_index: int,
    directory_path: str,
    files: list,
    repo_path: str,
    project_id: str,
    user_id: str,
    repo_name: str,
    commit_id: str = None,
    work_unit_db_id: str = None  # UUID of work unit in DB (for bootstrap/resume path only)
) -> Dict[str, Any]:
    """
    Worker task to parse a single directory work unit.

    Process:
    1. Create RepoMap instance for tree-sitter parsing
    2. Parse all files in parallel using ParallelFileParser
    3. Resolve intra-directory references
    4. Write nodes and edges to Neo4j incrementally
    5. Return defines/references for cross-directory resolution

    Args:
        work_unit_index: Index of this work unit (for logging)
        directory_path: Relative path of directory being parsed
        files: List of file paths to parse
        repo_path: Absolute path to repository root
        project_id: Project ID
        user_id: User ID
        repo_name: Repository name

    Returns:
        Dictionary with results
    """
    from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
    from app.modules.parsing.graph_construction.parallel_file_parser import ParallelFileParser
    from app.modules.parsing.graph_construction.code_graph_service import SimpleIO, SimpleTokenCounter
    from app.core.config_provider import config_provider
    from app.modules.parsing.graph_construction.neo4j_state_service import Neo4jStateService
    from app.modules.parsing.parsing_file_state_model import ParsingFileState
    from app.modules.parsing.parsing_work_unit_model import ParsingWorkUnit
    from datetime import datetime
    import time

    logger.info(
        f"[Unit {work_unit_index}] Starting: {directory_path or 'root'} "
        f"({len(files)} files)"
    )
    start_time = time.time()

    try:
        # Get Neo4j config (needed for both paths)
        neo4j_config = config_provider.get_neo4j_config()

        # Step 1: Check if we're in bootstrap/resume mode (work unit exists in DB)
        already_parsed = set()
        files_to_parse = files  # Default: parse all files

        if work_unit_db_id and commit_id:
            # Bootstrap/resume path: Check Neo4j and create file states
            neo4j_state_service = Neo4jStateService(
                neo4j_uri=neo4j_config["uri"],
                neo4j_user=neo4j_config["username"],
                neo4j_password=neo4j_config["password"]
            )

            try:
                # Query Neo4j for files that are already parsed (batched query)
                already_parsed = neo4j_state_service.get_parsed_files_for_paths(
                    project_id=project_id,
                    file_paths=files,
                    batch_size=1000
                )

                # Filter to only parse files that don't exist in Neo4j
                files_to_parse = [f for f in files if f not in already_parsed]

                logger.info(
                    f"[Unit {work_unit_index}] Neo4j check: {len(already_parsed)} already parsed, "
                    f"{len(files_to_parse)} need parsing"
                )

            finally:
                neo4j_state_service.close()

            # Step 2: Upsert file state records (idempotent - safe for retries)
            if commit_id:
                from sqlalchemy.dialects.postgresql import insert

                # Prepare file state data
                file_states_data = []
                for file_path in files:
                    file_status = 'completed' if file_path in already_parsed else 'pending'
                    file_state_data = {
                        'project_id': project_id,
                        'commit_id': commit_id,
                        'work_unit_id': work_unit_db_id,
                        'file_path': file_path,
                        'status': file_status,
                        'created_at': datetime.utcnow()
                    }
                    if file_status == 'completed':
                        file_state_data['processed_at'] = datetime.utcnow()
                    file_states_data.append(file_state_data)

                # Upsert using ON CONFLICT DO NOTHING (idempotent)
                if file_states_data:
                    stmt = insert(ParsingFileState.__table__).values(file_states_data)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=['project_id', 'commit_id', 'file_path']
                    )
                    self.db.execute(stmt)
                    self.db.commit()
                    logger.info(f"[Unit {work_unit_index}] Upserted {len(file_states_data)} file state records")

            # If all files already parsed, skip parsing
            if not files_to_parse:
                logger.info(f"[Unit {work_unit_index}] All files already parsed, skipping")

                # Update work unit status to completed
                work_unit = self.db.query(ParsingWorkUnit).filter(
                    ParsingWorkUnit.id == work_unit_db_id
                ).first()
                if work_unit:
                    work_unit.status = 'completed'
                    work_unit.completed_at = datetime.utcnow()
                    work_unit.nodes_created = 0
                    work_unit.edges_created = 0
                    self.db.commit()

                return {
                    'success': True,
                    'work_unit_index': work_unit_index,
                    'directory_path': directory_path,
                    'files_processed': 0,
                    'nodes_created': 0,
                    'edges_created': 0,
                    'immediate_edges': 0,
                    'deferred_references': 0,
                    'defines_count': 0,
                    'duration_seconds': time.time() - start_time,
                    'skipped': True,
                    'reason': 'all_files_already_parsed'
                }
        else:
            # Normal fresh parsing path: no Neo4j check, no file states
            logger.info(f"[Unit {work_unit_index}] Normal parsing mode (no file state tracking)")

        # Step 3: Initialize services for parsing
        repo_map = RepoMap(
            root=repo_path,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )

        parallel_parser = ParallelFileParser(
            repo_map_instance=repo_map,
            repo_path=repo_path,
            max_workers=15  # 15 threads per worker
        )

        # Step 4: Parse files in parallel (only unparsed files)
        G, defines, references = parallel_parser.parse_files_parallel(files_to_parse)

        logger.info(
            f"[Unit {work_unit_index}] Parsed {len(files_to_parse)} files: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        # Step 5: Resolve intra-directory references
        intra_refs_created = _resolve_intra_directory_references(
            G, defines, references
        )

        logger.info(
            f"[Unit {work_unit_index}] Resolved {intra_refs_created} "
            f"intra-directory references"
        )

        # Step 6: Write graph to Neo4j incrementally using CodeGraphService
        from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService

        code_graph_service = CodeGraphService(
            neo4j_uri=neo4j_config["uri"],
            neo4j_user=neo4j_config["username"],
            neo4j_password=neo4j_config["password"],
            db=self.db
        )

        try:
            nodes_created, edges_created = code_graph_service.write_subgraph_incremental(
                graph=G,
                project_id=project_id,
                user_id=user_id,
                batch_size=1000
            )

            # Step 7: Update file state records to completed (bulk update for performance)
            if work_unit_db_id and commit_id and files_to_parse:
                # Use bulk update with IN clause instead of individual updates
                self.db.query(ParsingFileState).filter(
                    ParsingFileState.project_id == project_id,
                    ParsingFileState.commit_id == commit_id,
                    ParsingFileState.work_unit_id == work_unit_db_id,
                    ParsingFileState.file_path.in_(files_to_parse)
                ).update({
                    'status': 'completed',
                    'processed_at': datetime.utcnow()
                }, synchronize_session=False)
                self.db.commit()
                logger.info(f"[Unit {work_unit_index}] Bulk updated {len(files_to_parse)} file states to completed")

            # Step 8: Try to immediately resolve cross-directory references
            # This is the hybrid optimization - resolve what we can now!
            logger.info(
                f"[Unit {work_unit_index}] Attempting immediate resolution of "
                f"{len(references)} cross-directory references"
            )

            immediate_edges, unresolved_refs = _resolve_references_immediately(
                code_graph_service=code_graph_service,
                references=references,
                project_id=project_id,
                work_unit_index=work_unit_index
            )

            logger.info(
                f"[Unit {work_unit_index}] Immediately resolved {immediate_edges} references, "
                f"{len(unresolved_refs)} deferred for later"
            )

        finally:
            code_graph_service.close()

        # Step 9: Update work unit status to completed (for bootstrap/resume path)
        if work_unit_db_id:
            work_unit = self.db.query(ParsingWorkUnit).filter(
                ParsingWorkUnit.id == work_unit_db_id
            ).first()
            if work_unit:
                work_unit.status = 'completed'
                work_unit.completed_at = datetime.utcnow()
                work_unit.nodes_created = nodes_created
                work_unit.edges_created = edges_created + immediate_edges
                self.db.commit()
                logger.info(f"[Unit {work_unit_index}] Marked work unit as completed in database")

        # Step 10: Check if this is the last worker to complete
        if work_unit_db_id and commit_id:
            from app.celery.coordination import ParsingCoordinator
            from app.modules.parsing.parsing_session_model import ParsingSession

            # Get total work units from session
            session = self.db.query(ParsingSession).filter(
                ParsingSession.project_id == project_id,
                ParsingSession.commit_id == commit_id,
                ParsingSession.completed_at.is_(None)
            ).first()

            if session:
                redis_client = self.app.backend.client
                completed_count, is_last = ParsingCoordinator.increment_completed(
                    redis_client,
                    project_id,
                    session.total_work_units
                )

                logger.info(
                    f"[Unit {work_unit_index}] Completion tracking: "
                    f"{completed_count}/{session.total_work_units}"
                )

                if is_last:
                    logger.info(
                        f"[Unit {work_unit_index}] Last worker completed - triggering finalization"
                    )
                    # Trigger finalization asynchronously with delay to ensure all DB writes complete
                    finalize_parsing.apply_async(
                        kwargs={
                            'project_id': project_id,
                            'user_id': user_id,
                            'repo_path': repo_path,
                            'commit_id': commit_id
                        },
                        countdown=5  # 5 second delay
                    )

        elapsed = time.time() - start_time
        logger.info(
            f"[Unit {work_unit_index}] Complete: {len(files_to_parse)} files parsed "
            f"({len(already_parsed)} skipped) in {elapsed:.2f}s "
            f"({nodes_created} nodes, {edges_created + immediate_edges} edges total)"
        )

        return {
            'success': True,
            'work_unit_index': work_unit_index,
            'directory_path': directory_path,
            'files_processed': len(files_to_parse),
            'files_skipped': len(already_parsed),
            'nodes_created': nodes_created,
            'edges_created': edges_created + immediate_edges,
            'immediate_edges': immediate_edges,
            'deferred_references': len(unresolved_refs),
            'defines_count': len(defines),
            'duration_seconds': elapsed
        }

    except Exception as e:
        logger.exception(
            f"[Unit {work_unit_index}] Error processing {directory_path}"
        )
        return {
            'success': False,
            'work_unit_index': work_unit_index,
            'directory_path': directory_path,
            'error': str(e),
            'files_processed': 0,
            'nodes_created': 0,
            'edges_created': 0,
            'immediate_edges': 0,
            'deferred_references': 0,
            'defines_count': 0
        }


def _resolve_intra_directory_references(G, defines, references) -> int:
    """
    Resolve references within this directory's subgraph.

    Returns:
        Number of REFERENCES edges created
    """
    from app.modules.parsing.graph_construction.parsing_repomap import RepoMap

    edges_created = 0
    seen_relationships = set()

    for ref in references:
        source = ref['source']
        target_ident = ref['target_ident']

        # Check if target defined in this subgraph
        target_nodes = defines.get(target_ident, set())

        for target in target_nodes:
            if source == target:
                continue

            if G.has_node(source) and G.has_node(target):
                RepoMap.create_relationship(
                    G,
                    source,
                    target,
                    'REFERENCES',
                    seen_relationships,
                    {
                        'ident': target_ident,
                        'ref_line': ref['line'],
                        'end_ref_line': ref['end_line']
                    }
                )
                edges_created += 1

    return edges_created


def _resolve_references_immediately(
    code_graph_service,
    references: list,
    project_id: str,
    work_unit_index: int
) -> tuple[int, list]:
    """
    Hybrid optimization: Try to resolve cross-directory references immediately.

    For each reference, query Neo4j to see if the target node exists.
    If it does, create the edge now. If not, defer for later resolution.

    This is the key optimization - most references can be resolved immediately
    because their targets were parsed by earlier workers.

    Args:
        code_graph_service: Neo4j service instance
        references: List of reference dicts
        project_id: Project ID
        work_unit_index: Worker index (for logging)

    Returns:
        Tuple of (edges_created, unresolved_references)
    """

    if not references:
        return 0, []

    edges_created = 0
    unresolved_refs = []
    batch_size = 100  # Process in small batches for progress tracking

    logger.info(f"[Unit {work_unit_index}] Attempting immediate resolution in batches of {batch_size}")

    driver = code_graph_service.driver

    with driver.session() as session:
        for i in range(0, len(references), batch_size):
            batch = references[i:i + batch_size]

            for ref in batch:
                source = ref['source']
                target_ident = ref['target_ident']

                # Try to create the edge if target exists
                # Use MERGE to avoid duplicate edges
                result = session.run(
                    """
                    MATCH (source:NODE {name: $source_name, repoId: $project_id})
                    MATCH (target:NODE {name: $target_ident, repoId: $project_id})
                    MERGE (source)-[r:REFERENCES {
                        repoId: $project_id,
                        ident: $target_ident,
                        ref_line: $ref_line,
                        end_ref_line: $end_ref_line
                    }]->(target)
                    RETURN count(r) as created
                    """,
                    source_name=source,
                    target_ident=target_ident,
                    project_id=project_id,
                    ref_line=ref.get('line'),
                    end_ref_line=ref.get('end_line')
                )

                record = result.single()
                if record and record['created'] > 0:
                    edges_created += 1
                else:
                    # Target doesn't exist yet, defer for later
                    unresolved_refs.append(ref)

            # Log progress every batch
            if (i + batch_size) % 1000 == 0:
                logger.info(
                    f"[Unit {work_unit_index}] Processed {i + batch_size}/{len(references)} refs, "
                    f"resolved {edges_created}, deferred {len(unresolved_refs)}"
                )

    return edges_created, unresolved_refs


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.finalize_parsing",
    time_limit=1800,  # 30 minutes
    ignore_result=False,
)
def finalize_parsing(
    self,
    project_id: str,
    user_id: str,
    repo_path: str = None,
    commit_id: str = None
) -> Dict[str, Any]:
    """
    Finalize parsing by querying aggregated stats from database.

    Triggered by last worker to complete (no chord callback needed).
    This replaces aggregate_and_resolve_references to avoid OOM from loading
    hundreds of task results into memory.

    Args:
        project_id: Project ID
        user_id: User ID
        repo_path: Path to cloned repository (for cleanup)
        commit_id: Commit ID for this parsing session

    Returns:
        Dictionary with finalization results
    """
    import time
    import shutil
    from app.modules.parsing.parsing_work_unit_model import ParsingWorkUnit
    from app.modules.parsing.parsing_session_model import ParsingSession
    from datetime import datetime

    logger.info(f"Finalizing parsing for project {project_id}")
    start_time = time.time()

    try:
        # Check if already finalized (idempotency)
        session = self.db.query(ParsingSession).filter(
            ParsingSession.project_id == project_id,
            ParsingSession.commit_id == commit_id
        ).first()

        if session and session.completed_at:
            logger.info(f"Project {project_id} already finalized at {session.completed_at}, skipping")
            return {
                'success': True,
                'project_id': project_id,
                'status': 'already_completed',
                'completed_at': session.completed_at.isoformat()
            }

        # Query work units from database for aggregation
        work_units = self.db.query(ParsingWorkUnit).filter(
            ParsingWorkUnit.project_id == project_id,
            ParsingWorkUnit.commit_id == commit_id
        ).all()

        if not work_units:
            logger.warning(f"No work units found for project {project_id}, commit {commit_id}")
            return {
                'success': False,
                'error': 'No work units found',
                'project_id': project_id
            }

        # Aggregate stats from database
        total_nodes = sum(u.nodes_created or 0 for u in work_units)
        total_edges = sum(u.edges_created or 0 for u in work_units)
        total_files = sum(u.file_count or 0 for u in work_units)

        completed_units = [u for u in work_units if u.status == 'completed']
        failed_units = [u for u in work_units if u.status == 'failed']

        logger.info(
            f"Aggregation complete: {len(completed_units)}/{len(work_units)} units succeeded, "
            f"{total_files} files, {total_nodes} nodes, {total_edges} edges"
        )

        if failed_units:
            logger.warning(f"{len(failed_units)} work units failed")

        # Spawn inference if enabled
        enable_inference = os.getenv("ENABLE_INFERENCE", "false").lower() == "true"

        if enable_inference:
            try:
                inference_result = spawn_inference_chord(self, project_id, user_id)
                logger.info(
                    f"Spawned inference chord for project {project_id}: "
                    f"{inference_result['work_units']} inference units"
                )
            except Exception as inference_error:
                error_msg = str(inference_error)
                logger.exception(
                    f"Failed to spawn inference chord for project {project_id}: {error_msg}"
                )
                # Mark project as ERROR since inference is critical
                from app.modules.projects.projects_service import ProjectService
                from app.modules.projects.projects_schema import ProjectStatusEnum

                try:
                    project_service = ProjectService(self.db)

                    async def mark_error():
                        logger.error(f"Inference setup failed: {error_msg}")
                        await project_service.update_project_status(
                            project_id=project_id,
                            status=ProjectStatusEnum.ERROR,
                        )

                    self.run_async(mark_error())
                except Exception as status_error:
                    logger.exception(f"Failed to update project status: {status_error}")

                # Cleanup: Remove cloned repository before returning on error
                cleanup_success = False
                if repo_path and os.path.exists(repo_path):
                    try:
                        logger.info(f"Cleaning up cloned repository at: {repo_path} (inference failure)")
                        shutil.rmtree(repo_path)
                        cleanup_success = True
                        logger.info(f"Successfully cleaned up repository: {repo_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup repository {repo_path}: {cleanup_error}")

                return {
                    'success': False,
                    'error': error_msg,
                    'project_id': project_id,
                    'repo_cleanup': cleanup_success
                }
        else:
            # Inference is disabled, update project status to READY directly
            logger.info(f"Skipping inference for project {project_id} (ENABLE_INFERENCE=false)")
            from app.modules.projects.projects_service import ProjectService
            from app.modules.projects.projects_schema import ProjectStatusEnum

            try:
                project_service = ProjectService(self.db)

                async def mark_ready():
                    await project_service.update_project_status(
                        project_id=project_id,
                        status=ProjectStatusEnum.READY,
                    )
                    logger.info(f"Project {project_id} status updated to READY (inference skipped)")

                self.run_async(mark_ready())
            except Exception as status_error:
                logger.exception(f"Failed to update project status to READY: {status_error}")

        # Cleanup: Remove cloned repository after all workers are done
        cleanup_success = False
        if repo_path and os.path.exists(repo_path):
            try:
                logger.info(f"Cleaning up cloned repository at: {repo_path}")
                shutil.rmtree(repo_path)
                cleanup_success = True
                logger.info(f"Successfully cleaned up repository: {repo_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup repository {repo_path}: {cleanup_error}")

        # Mark session as complete
        if session:
            session.completed_at = datetime.utcnow()
            session.stage = 'completed'
            self.db.commit()
            logger.info(f"Marked session {session.id} as completed")

        elapsed = time.time() - start_time

        # Build return dict based on whether inference was enabled
        result = {
            'success': len(failed_units) == 0,
            'project_id': project_id,
            'total_files': total_files,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'work_units': len(work_units),
            'failed_units': len(failed_units),
            'duration_seconds': elapsed,
            'inference_spawned': enable_inference,
            'repo_cleanup': cleanup_success
        }

        if enable_inference:
            result['inference_units'] = inference_result.get('work_units', 0)
            result['status'] = 'running_inference'
        else:
            result['status'] = 'ready'
            result['inference_skipped'] = True

        return result

    except Exception as e:
        logger.exception(f"Error finalizing parsing for project {project_id}")

        # Best effort cleanup even on error
        if repo_path and os.path.exists(repo_path):
            try:
                logger.info(f"Cleaning up repository on error: {repo_path}")
                shutil.rmtree(repo_path)
            except Exception:
                pass

        return {
            'success': False,
            'error': str(e),
            'project_id': project_id
        }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.aggregate_and_resolve_references",
    time_limit=1800,  # 30 minutes
    ignore_result=False,  # Store results for monitoring and inference coordination
)
def aggregate_and_resolve_references(
    self,
    task_results: list,
    project_id: str,
    user_id: str,
    total_work_units: int,
    start_time: float,
    repo_path: str = None  # Optional for backward compatibility
) -> Dict[str, Any]:
    """
    DEPRECATED: Replaced by finalize_parsing + Redis coordination.

    This callback caused OOM with large chord results (811+ tasks).
    Now using Redis atomic counters instead of chord callbacks.

    This function is kept for backward compatibility with existing
    parsing sessions but should not be called for new sessions.
    """
    logger.warning(
        f"DEPRECATED: aggregate_and_resolve_references called for project {project_id}. "
        f"This function is deprecated and replaced by finalize_parsing. "
        f"Received {len(task_results)} task results."
    )

    # For backward compatibility, if this gets called, try to finalize properly
    # by delegating to the new finalize_parsing function
    logger.info(f"Delegating to finalize_parsing for project {project_id}")

    return finalize_parsing(
        self,
        project_id=project_id,
        user_id=user_id,
        repo_path=repo_path,
        commit_id=None  # We don't have it in old sessions
    )


def spawn_inference_chord(task_instance, project_id: str, user_id: str) -> Dict[str, Any]:
    """
    Helper function to spawn inference chord with work units.

    This function:
    1. Queries Neo4j to get directory structure and node counts
    2. Builds work units with dynamic splitting for large directories
    3. Creates a Celery chord with run_inference_unit workers
    4. Sets finalize_project_after_inference as callback

    Args:
        task_instance: Celery task instance (for db access)
        project_id: Project ID
        user_id: User ID

    Returns:
        Dict with chord info (work_units count, etc.)
    """
    from celery import chord
    from app.core.config_provider import config_provider
    from neo4j import GraphDatabase

    logger.info(f"Building inference work units for project {project_id}")

    # Initialize driver to None for proper cleanup
    driver = None

    try:
        # Connect to Neo4j
        neo4j_config = config_provider.get_neo4j_config()
        driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

        with driver.session() as session:
            # Query to get node counts per directory
            result = session.run("""
                MATCH (n:NODE {repoId: $repo_id})
                WHERE n.file_path IS NOT NULL AND n.file_path <> ''
                WITH n.file_path AS file_path
                WITH
                    CASE
                        WHEN file_path CONTAINS '/'
                        THEN split(file_path, '/')[0..-1]
                        ELSE []
                    END AS path_parts
                WITH
                    CASE
                        WHEN size(path_parts) = 0 THEN '<ROOT>'
                        ELSE reduce(p = '', part IN path_parts | p + '/' + part)
                    END AS directory
                RETURN directory, count(*) AS node_count
                ORDER BY node_count DESC
            """, repo_id=project_id)

            directory_counts = {
                record["directory"]: record["node_count"]
                for record in result
            }

        logger.info(
            f"Found {len(directory_counts)} directories for project {project_id}"
        )

        # Build work units with dynamic splitting (no driver needed here, query complete)
        work_units = []
        max_nodes_per_unit = int(os.getenv('MAX_INFERENCE_NODES_PER_UNIT', '2000'))

        for directory, node_count in directory_counts.items():
            is_root = (directory == '<ROOT>')

            if node_count > max_nodes_per_unit:
                # Split large directories into multiple units
                num_splits = (node_count // max_nodes_per_unit) + 1
                logger.info(
                    f"Splitting directory '{directory}' ({node_count} nodes) "
                    f"into {num_splits} units"
                )

                for split_index in range(num_splits):
                    work_units.append({
                        'project_id': project_id,
                        'user_id': user_id,
                        'directory_path': None if is_root else directory.lstrip('/'),
                        'is_root': is_root,
                        'split_index': split_index,
                        'total_splits': num_splits,
                    })
            else:
                # Single unit for this directory
                work_units.append({
                    'project_id': project_id,
                    'user_id': user_id,
                    'directory_path': None if is_root else directory.lstrip('/'),
                    'is_root': is_root,
                    'split_index': None,
                    'total_splits': None,
                })

        logger.info(
            f"Created {len(work_units)} inference work units for project {project_id}"
        )

        # Create chord: workers + callback
        inference_chord = chord(
            [
                run_inference_unit.signature(
                    kwargs=unit,
                    queue=os.getenv('CELERY_QUEUE_NAME', 'staging') + '_process_repository'
                )
                for unit in work_units
            ]
        )(
            finalize_project_after_inference.signature(
                kwargs={
                    'project_id': project_id,
                    'user_id': user_id,
                },
                queue=os.getenv('CELERY_QUEUE_NAME', 'staging') + '_process_repository'
            )
        )

        logger.info(
            f"Spawned inference chord for project {project_id}: "
            f"{len(work_units)} workers, chord_id={inference_chord.id}"
        )

        return {
            'success': True,
            'work_units': len(work_units),
            'chord_id': inference_chord.id,
        }

    except Exception as e:
        logger.exception(f"Error building inference work units: {e}")
        raise

    finally:
        # Always close driver if it was created
        if driver is not None:
            try:
                driver.close()
                logger.debug(f"Closed Neo4j driver for project {project_id}")
            except Exception as close_error:
                logger.error(f"Error closing Neo4j driver: {close_error}")


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.resolve_cross_directory_references",
    time_limit=600,  # 10 minutes
    ignore_result=False,  # Store results for monitoring
)
def resolve_cross_directory_references(
    self,
    project_id: str,
    user_id: str,
    all_defines: Dict[str, list],
    all_references: list
) -> Dict[str, Any]:
    """
    Resolve references across directory boundaries.

    This task runs after all directory workers complete. It:
    1. Receives aggregated defines and references from all workers
    2. Matches references to definitions in batches
    3. Creates REFERENCES edges in Neo4j incrementally

    Memory optimization: Processes references in batches and streams edges to Neo4j
    instead of building the entire edge list in memory.

    Args:
        project_id: Project ID
        user_id: User ID
        all_defines: Aggregated defines from all workers
            Format: {identifier: [node_name1, node_name2, ...]}
        all_references: Aggregated references from all workers
            Format: [{source, target_ident, line, end_line, ...}, ...]

    Returns:
        Dictionary with results
    """
    from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
    from app.core.config_provider import config_provider
    import time

    logger.info(
        f"Resolving cross-directory references for project {project_id}: "
        f"{len(all_defines)} identifiers, {len(all_references)} references"
    )
    start_time = time.time()

    try:
        # Convert defines lists back to sets for efficient lookup
        defines = {
            ident: set(nodes) for ident, nodes in all_defines.items()
        }

        # Setup Neo4j connection once
        neo4j_config = config_provider.get_neo4j_config()
        code_graph_service = CodeGraphService(
            neo4j_uri=neo4j_config["uri"],
            neo4j_user=neo4j_config["username"],
            neo4j_password=neo4j_config["password"],
            db=self.db
        )

        # Process references in batches to avoid memory issues
        edge_batch_size = int(os.getenv('NEO4J_BATCH_SIZE_DISTRIBUTED', '1000'))

        total_edges_created = 0
        seen_relationships = set()
        edges_to_create = []
        processed_refs = 0

        try:
            for ref in all_references:
                source = ref['source']
                target_ident = ref['target_ident']

                # Find matching definitions
                target_nodes = defines.get(target_ident, set())

                for target in target_nodes:
                    if source == target:
                        continue

                    # Create edge dict
                    rel_key = (source, target, 'REFERENCES')
                    if rel_key not in seen_relationships:
                        edges_to_create.append({
                            'source': source,
                            'target': target,
                            'type': 'REFERENCES',
                            'ident': target_ident,
                            'ref_line': ref['line'],
                            'end_ref_line': ref['end_line'],
                        })
                        seen_relationships.add(rel_key)

                processed_refs += 1

                # Write edges to Neo4j when we hit the batch size
                if len(edges_to_create) >= edge_batch_size:
                    edges_created = code_graph_service.create_edges_batch(
                        edges_list=edges_to_create,
                        project_id=project_id,
                        user_id=user_id,
                        batch_size=edge_batch_size
                    )
                    total_edges_created += edges_created
                    logger.info(
                        f"Wrote {edges_created} edges to Neo4j "
                        f"(processed {processed_refs}/{len(all_references)} refs, "
                        f"total edges: {total_edges_created})"
                    )
                    edges_to_create = []  # Clear batch

            # Write any remaining edges
            if edges_to_create:
                edges_created = code_graph_service.create_edges_batch(
                    edges_list=edges_to_create,
                    project_id=project_id,
                    user_id=user_id,
                    batch_size=edge_batch_size
                )
                total_edges_created += edges_created
                logger.info(f"Wrote final {edges_created} edges to Neo4j")

        finally:
            code_graph_service.close()

        elapsed = time.time() - start_time
        logger.info(
            f"Cross-directory reference resolution complete: "
            f"{total_edges_created} edges created from {len(all_references)} references in {elapsed:.2f}s"
        )

        # INTEGRATION: Spawn inference chord after reference resolution
        # This is where we integrate the missing inference step
        try:
            inference_chord_result = spawn_inference_chord(
                self, project_id, user_id
            )
            logger.info(
                f"Spawned inference chord for project {project_id}: "
                f"{inference_chord_result['work_units']} inference units"
            )

            return {
                'success': True,
                'edges_created': total_edges_created,
                'duration_seconds': elapsed,
                'inference_spawned': True,
                'inference_units': inference_chord_result['work_units'],
            }
        except Exception as inference_error:
            error_msg = str(inference_error)
            logger.exception(
                f"Failed to spawn inference chord for project {project_id}: {error_msg}"
            )
            # Mark project as ERROR since inference is critical
            from app.modules.projects.projects_service import ProjectService
            from app.modules.projects.projects_schema import ProjectStatusEnum

            try:
                project_service = ProjectService(self.db)

                async def mark_error():
                    logger.error(f"Inference setup failed: {error_msg}")
                    await project_service.update_project_status(
                        project_id=project_id,
                        status=ProjectStatusEnum.ERROR,
                    )

                self.run_async(mark_error())
            except Exception as status_error:
                logger.exception(f"Failed to update project status: {status_error}")

            return {
                'success': False,
                'edges_created': total_edges_created,
                'duration_seconds': elapsed,
                'inference_spawned': False,
                'error': error_msg,
            }

    except Exception as e:
        logger.exception("Error resolving cross-directory references")
        return {
            'success': False,
            'error': str(e),
            'edges_created': 0
        }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.run_inference_unit",
    time_limit=3600,  # 1 hour
    ignore_result=False,  # Results must be stored for inference chord callback
)
def run_inference_unit(
    self,
    project_id: str,
    user_id: str,
    directory_path: str = None,
    is_root: bool = False,
    split_index: int = None,
    total_splits: int = None,
) -> Dict[str, Any]:
    """
    Worker task for running inference on a directory or directory split.

    Uses Redis distributed semaphore for global rate limiting and processes
    nodes in streaming fashion to avoid OOM.

    Args:
        project_id: Project ID
        user_id: User ID
        directory_path: Directory to process (None = entire repo)
        is_root: Whether this is the root directory
        split_index: Index of split for large directories (0-based)
        total_splits: Total number of splits for this directory

    Returns:
        Dict with results (nodes_processed, batches_processed, etc.)
    """
    from app.modules.parsing.knowledge_graph.inference_service import InferenceService
    from app.celery.redis_semaphore import get_redis_semaphore
    import time

    unit_desc = (
        "root" if is_root
        else f"directory '{directory_path}'" if directory_path and not split_index
        else f"directory '{directory_path}' split {split_index + 1}/{total_splits}"
        if split_index is not None
        else "entire repo"
    )

    logger.info(
        f"Starting inference for project {project_id}, {unit_desc}"
    )
    start_time = time.time()

    try:
        inference_service = InferenceService(self.db, user_id)

        # Get Redis semaphore for global rate limiting
        semaphore = get_redis_semaphore(
            key_suffix=f"inference:{project_id}",
            max_concurrent=int(os.getenv("MAX_GLOBAL_LLM_REQUESTS", "50")),
            ttl=300,  # 5 minutes
        )

        async def run_inference_with_semaphore():
            """
            Run inference with global rate limiting via Redis semaphore.

            This wraps the entire process_nodes_streaming call with semaphore
            to ensure we don't exceed MAX_GLOBAL_LLM_REQUESTS across all workers.
            """
            # Acquire semaphore for the duration of processing
            async with semaphore.acquire(timeout=120):
                logger.info(
                    f"Acquired global semaphore for {unit_desc} "
                    f"(current: {semaphore.get_current_count()}/{semaphore.max_concurrent})"
                )

                result = await inference_service.process_nodes_streaming(
                    repo_id=project_id,
                    directory_path=directory_path,
                    is_root=is_root,
                    chunk_size=500,
                )

                return result

        result = self.run_async(run_inference_with_semaphore())
        inference_service.close()

        elapsed = time.time() - start_time
        logger.info(
            f"Inference complete for {unit_desc}: "
            f"{result['total_nodes_processed']} nodes processed in {elapsed:.2f}s"
        )

        return {
            'success': True,
            'directory_path': directory_path,
            'is_root': is_root,
            'split_index': split_index,
            'total_splits': total_splits,
            'nodes_processed': result['total_nodes_processed'],
            'batches_processed': result['total_batches_processed'],
            'nodes_indexed': result['nodes_indexed'],
            'duration_seconds': elapsed,
        }

    except Exception as e:
        logger.exception(
            f"Error during inference for {unit_desc}"
        )
        return {
            'success': False,
            'directory_path': directory_path,
            'is_root': is_root,
            'split_index': split_index,
            'error': str(e),
            'nodes_processed': 0,
        }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.finalize_project_after_inference",
    time_limit=600,  # 10 minutes
    ignore_result=False,  # Store results for monitoring and status tracking
)
def finalize_project_after_inference(
    self,
    inference_results: list,
    project_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Callback task after all inference units complete.

    This task:
    1. Aggregates results from all workers
    2. Commits search indices (single transaction)
    3. Spawns async vector index creation task
    4. Updates project status based on completion rate
    5. Handles partial failures gracefully

    Args:
        inference_results: List of results from run_inference_unit tasks
        project_id: Project ID
        user_id: User ID

    Returns:
        Dict with finalization results
    """
    from app.modules.projects.projects_service import ProjectService
    from app.modules.search.search_service import SearchService
    from app.modules.projects.projects_schema import ProjectStatusEnum
    import time

    logger.info(
        f"Finalizing project {project_id} after inference: "
        f"{len(inference_results)} units processed"
    )
    start_time = time.time()

    try:
        # Aggregate results
        successful_units = [r for r in inference_results if r.get('success')]
        failed_units = [r for r in inference_results if not r.get('success')]

        total_nodes_processed = sum(
            r.get('nodes_processed', 0) for r in successful_units
        )
        total_batches_processed = sum(
            r.get('batches_processed', 0) for r in successful_units
        )

        logger.info(
            f"Inference aggregation: {len(successful_units)}/{len(inference_results)} "
            f"units succeeded, {total_nodes_processed} total nodes processed"
        )

        # Calculate completion rate with explicit edge case handling
        if not inference_results or len(inference_results) == 0:
            completion_rate = 0.0
            logger.warning(
                f"No inference results to calculate completion rate for project {project_id}"
            )
        elif len(successful_units) == 0:
            completion_rate = 0.0
            logger.error(
                f"All inference units failed for project {project_id}: "
                f"0/{len(inference_results)} succeeded"
            )
        else:
            completion_rate = len(successful_units) / len(inference_results)
            logger.info(
                f"Completion rate for project {project_id}: "
                f"{completion_rate:.1%} ({len(successful_units)}/{len(inference_results)})"
            )

        # Commit search indices (single transaction to avoid contention)
        search_service = SearchService(self.db)

        async def commit_search_indices():
            await search_service.commit_indices()
            logger.info(f"Search indices committed for project {project_id}")

        self.run_async(commit_search_indices())

        # Spawn async vector index creation task (don't block on it)
        create_vector_index_async.apply_async(
            args=[project_id, user_id],
            priority=5,  # Lower priority
        )
        logger.info(f"Spawned vector index creation task for project {project_id}")

        # Determine final status based on completion rate
        project_service = ProjectService(self.db)

        if completion_rate >= 0.95:
            # 95%+ success - mark as READY (with warning if not 100%)
            final_status = ProjectStatusEnum.READY
            status_message = (
                "Inference completed successfully"
                if completion_rate == 1.0
                else f"Inference completed with {len(failed_units)} failed units"
            )
        elif completion_rate >= 0.75:
            # 75-95% success - mark as PARTIALLY_READY
            final_status = ProjectStatusEnum.PARTIALLY_READY
            status_message = (
                f"Inference partially completed: {len(successful_units)}/{len(inference_results)} "
                f"units succeeded. Some functionality may be limited."
            )
        else:
            # <75% success - mark as ERROR
            final_status = ProjectStatusEnum.ERROR
            status_message = (
                f"Inference failed: only {len(successful_units)}/{len(inference_results)} "
                f"units succeeded"
            )

        # Update project status
        async def update_project_status():
            # Log status message for tracking
            if final_status != ProjectStatusEnum.READY:
                logger.warning(f"Project {project_id}: {status_message}")

            await project_service.update_project_status(
                project_id=project_id,
                status=final_status,
            )
            logger.info(
                f"Project {project_id} status updated to {final_status.value}: {status_message}"
            )

        self.run_async(update_project_status())

        elapsed = time.time() - start_time
        logger.info(
            f"Finalization complete for project {project_id} in {elapsed:.2f}s"
        )

        return {
            'success': True,
            'project_id': project_id,
            'final_status': final_status.value,
            'total_units': len(inference_results),
            'successful_units': len(successful_units),
            'failed_units': len(failed_units),
            'total_nodes_processed': total_nodes_processed,
            'total_batches_processed': total_batches_processed,
            'completion_rate': completion_rate,
            'duration_seconds': elapsed,
        }

    except Exception as e:
        logger.exception(f"Error finalizing project {project_id}")

        # Best effort: try to mark project as ERROR
        try:
            project_service = ProjectService(self.db)
            error_msg = str(e)

            async def mark_error():
                logger.error(f"Finalization failed: {error_msg}")
                await project_service.update_project_status(
                    project_id=project_id,
                    status=ProjectStatusEnum.ERROR,
                )

            self.run_async(mark_error())
        except Exception as inner_e:
            logger.exception(f"Failed to mark project {project_id} as ERROR: {inner_e}")

        return {
            'success': False,
            'error': str(e),
            'project_id': project_id,
        }


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.create_vector_index_async",
    time_limit=7200,  # 2 hours for very large repos
    ignore_result=False,  # Store results for monitoring
)
def create_vector_index_async(
    self,
    project_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Async task to create Neo4j vector index.

    This runs in the background to avoid blocking project finalization.
    For very large repos (500k+ nodes), index creation can take hours.

    Args:
        project_id: Project ID
        user_id: User ID

    Returns:
        Dict with results
    """
    from app.modules.parsing.knowledge_graph.inference_service import InferenceService
    import time

    logger.info(f"Creating vector index for project {project_id}")
    start_time = time.time()

    try:
        inference_service = InferenceService(self.db, user_id)
        inference_service.create_vector_index()
        inference_service.close()

        elapsed = time.time() - start_time
        logger.info(
            f"Vector index created for project {project_id} in {elapsed:.2f}s"
        )

        return {
            'success': True,
            'project_id': project_id,
            'duration_seconds': elapsed,
        }

    except Exception as e:
        logger.exception(f"Error creating vector index for project {project_id}")
        return {
            'success': False,
            'error': str(e),
            'project_id': project_id,
        }


logger.info("Parsing tasks module loaded")
