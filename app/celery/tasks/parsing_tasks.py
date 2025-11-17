import logging
import os
from typing import Any, Dict

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_parsing",
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


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_parsing_distributed",
    time_limit=7200,  # 2 hours
)
def process_parsing_distributed(
    self,
    repo_details: dict,
    user_id: str,
    user_email: str,
    project_id: str,
    cleanup_graph: bool = False
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

        # Setup project directory
        project_path, _ = self.run_async(setup_project_dir(repo, auth))
        logger.info(f"Project directory setup: {project_path}")

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
        parsing_tasks = []
        for i, work_unit in enumerate(work_units):
            task = parse_directory_unit.s(
                work_unit_index=i,
                directory_path=work_unit.path,
                files=work_unit.files,
                repo_path=project_path,
                project_id=project_id,
                user_id=user_id,
                repo_name=repo_details.get('repo_name', '')
            )
            parsing_tasks.append(task)

        # Step 4: Use chord to execute tasks in parallel with callback
        # The callback (aggregate_and_resolve) will be called after all parsing tasks complete
        callback = aggregate_and_resolve_references.s(
            project_id=project_id,
            user_id=user_id,
            total_work_units=len(work_units),
            start_time=start_time,
            repo_path=project_path  # For cleanup after all workers finish
        )

        # Execute chord: (group of tasks) | callback
        chord_task = chord(parsing_tasks)(callback)

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
            'chord_task_id': chord_task.id,
            'status': 'processing'
        }

    except Exception as e:
        logger.exception(f"Error in distributed parsing coordinator")
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
)
def parse_directory_unit(
    self,
    work_unit_index: int,
    directory_path: str,
    files: list,
    repo_path: str,
    project_id: str,
    user_id: str,
    repo_name: str
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
    from neo4j import GraphDatabase
    import time

    logger.info(
        f"[Unit {work_unit_index}] Starting: {directory_path or 'root'} "
        f"({len(files)} files)"
    )
    start_time = time.time()

    try:
        # Step 1: Initialize services
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

        # Step 2: Parse files in parallel
        G, defines, references = parallel_parser.parse_files_parallel(files)

        logger.info(
            f"[Unit {work_unit_index}] Parsed {len(files)} files: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        # Step 3: Resolve intra-directory references
        intra_refs_created = _resolve_intra_directory_references(
            G, defines, references
        )

        logger.info(
            f"[Unit {work_unit_index}] Resolved {intra_refs_created} "
            f"intra-directory references"
        )

        # Step 4: Write graph to Neo4j incrementally using CodeGraphService
        from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
        neo4j_config = config_provider.get_neo4j_config()

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

            # Step 5: Try to immediately resolve cross-directory references
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

        elapsed = time.time() - start_time
        logger.info(
            f"[Unit {work_unit_index}] Complete: {len(files)} files in {elapsed:.2f}s "
            f"({nodes_created} nodes, {edges_created + immediate_edges} edges total)"
        )

        # Step 6: Export defines and ONLY unresolved references for cross-directory resolution
        # Convert sets to lists for JSON serialization
        defines_serializable = {
            ident: list(node_set) for ident, node_set in defines.items()
        }

        return {
            'success': True,
            'work_unit_index': work_unit_index,
            'directory_path': directory_path,
            'files_processed': len(files),
            'nodes_created': nodes_created,
            'edges_created': edges_created + immediate_edges,
            'immediate_edges': immediate_edges,
            'deferred_references': len(unresolved_refs),
            'defines': defines_serializable,
            'references': unresolved_refs,  # Only unresolved ones!
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
            'defines': {},
            'references': []
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
    from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService

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
    name="app.celery.tasks.parsing_tasks.aggregate_and_resolve_references",
    time_limit=1800,  # 30 minutes
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
    Callback task that aggregates parsing results and triggers reference resolution.

    This is executed automatically by Celery chord after all parse_directory_unit tasks complete.

    Args:
        task_results: List of results from all parse_directory_unit tasks
        project_id: Project ID
        user_id: User ID
        total_work_units: Total number of work units processed
        start_time: Start time of the entire parsing operation
        repo_path: Path to cloned repository (for cleanup after all workers finish)

    Returns:
        Dictionary with final parsing results
    """
    import time
    import shutil
    from collections import defaultdict

    logger.info(
        f"Aggregating results from {len(task_results)} work units for project {project_id}"
    )

    try:
        # Aggregate results
        total_nodes_created = sum(r['nodes_created'] for r in task_results)
        total_edges_created = sum(r['edges_created'] for r in task_results)
        total_files_processed = sum(r['files_processed'] for r in task_results)
        total_immediate_edges = sum(r.get('immediate_edges', 0) for r in task_results)
        total_deferred_refs = sum(r.get('deferred_references', 0) for r in task_results)
        failed_units = [r for r in task_results if not r['success']]

        logger.info(
            f"Parallel parsing complete: "
            f"{total_files_processed} files, "
            f"{total_nodes_created} nodes, "
            f"{total_edges_created} edges "
            f"(including {total_immediate_edges} immediate cross-directory refs)"
        )

        if failed_units:
            logger.warning(f"{len(failed_units)} work units failed")

        # Collect all references from task results
        # IMPORTANT: With hybrid resolution, these are ONLY the unresolved references!
        # Most references were already resolved by workers
        all_defines = defaultdict(set)
        all_references = []
        for r in task_results:
            # Merge defines by unioning sets for each identifier
            for ident, node_list in r.get('defines', {}).items():
                # Convert list to set and union with existing
                all_defines[ident].update(node_list)
            all_references.extend(r.get('references', []))

        # Convert defaultdict back to regular dict with lists for serialization
        all_defines = {ident: list(node_set) for ident, node_set in all_defines.items()}

        logger.info(
            f"Hybrid resolution stats: "
            f"{total_immediate_edges} refs resolved immediately by workers, "
            f"{len(all_references)} deferred for final resolution "
            f"({len(all_defines)} unique identifiers)"
        )

        # Trigger reference resolution task asynchronously
        # This is safe because this callback task is separate from the coordinator
        resolve_task = resolve_cross_directory_references.apply_async(
            args=[project_id, user_id, all_defines, all_references]
        )

        elapsed = time.time() - start_time

        # Cleanup: Remove cloned repository after all workers are done
        # This is safe because all parsing tasks have completed (shared storage)
        cleanup_success = False
        if repo_path and os.path.exists(repo_path):
            try:
                logger.info(f"Cleaning up cloned repository at: {repo_path}")
                shutil.rmtree(repo_path)
                cleanup_success = True
                logger.info(f"Successfully cleaned up repository: {repo_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup repository {repo_path}: {cleanup_error}")
                # Don't fail the task due to cleanup errors

        return {
            'success': len(failed_units) == 0,
            'project_id': project_id,
            'total_files': total_files_processed,
            'total_nodes': total_nodes_created,
            'total_edges': total_edges_created,
            'immediate_edges': total_immediate_edges,
            'deferred_references': len(all_references),
            'work_units': total_work_units,
            'failed_units': len(failed_units),
            'duration_seconds': elapsed,
            'workers_used': total_work_units,
            'resolution_task_id': resolve_task.id,
            'status': 'resolving_references',
            'repo_cleanup': cleanup_success
        }

    except Exception as e:
        logger.exception("Error aggregating parsing results")

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
    from collections import defaultdict

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
        ref_batch_size = int(os.getenv('CROSS_REF_BATCH_SIZE', '10000'))
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
            logger.exception(
                f"Failed to spawn inference chord for project {project_id}: {inference_error}"
            )
            # Mark project as ERROR since inference is critical
            from app.modules.projects.projects_service import ProjectService
            from app.modules.projects.projects_schema import ProjectStatusEnum

            try:
                project_service = ProjectService(self.db)

                async def mark_error():
                    logger.error(f"Inference setup failed: {str(inference_error)}")
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
                'error': str(inference_error),
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
        f"root" if is_root
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

            async def mark_error():
                logger.error(f"Finalization failed: {str(e)}")
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
