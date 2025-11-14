import logging
import os
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import networkx as nx
import time

logger = logging.getLogger(__name__)


class ParallelFileParser:
    """
    Parses multiple files in parallel and builds a local subgraph.
    """

    def __init__(self, repo_map_instance, repo_path: str, max_workers: int = 15):
        """
        Args:
            repo_map_instance: Instance of RepoMap for tag extraction
            repo_path: Absolute path to repository root
            max_workers: Number of parallel threads for file parsing
        """
        self.repo_map = repo_map_instance
        self.repo_path = repo_path
        self.max_workers = max_workers

    def parse_files_parallel(
        self,
        file_list: List[str]
    ) -> Tuple[nx.MultiDiGraph, Dict[str, Set[str]], List[Dict[str, Any]]]:
        """
        Parse multiple files in parallel and build subgraph.

        Args:
            file_list: List of relative file paths to parse

        Returns:
            Tuple of:
            - NetworkX graph with nodes and edges
            - defines: Dict mapping identifier -> set of node names
            - references: List of reference dicts for cross-directory resolution
        """
        logger.info(f"Parsing {len(file_list)} files with {self.max_workers} workers")
        start_time = time.time()

        # Thread-safe collections
        G = nx.MultiDiGraph()
        defines = defaultdict(set)
        references = []
        seen_relationships = set()

        # Use lock for thread-safe graph modifications
        import threading
        graph_lock = threading.Lock()

        # Parse files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file parsing tasks
            future_to_file = {
                executor.submit(self._parse_single_file, rel_path): rel_path
                for rel_path in file_list
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                rel_path = future_to_file[future]
                try:
                    file_result = future.result()

                    if file_result:
                        # Merge results into main graph (thread-safe)
                        with graph_lock:
                            self._merge_file_result(
                                G=G,
                                file_result=file_result,
                                defines=defines,
                                references=references,
                                seen_relationships=seen_relationships
                            )

                    completed += 1
                    if completed % 100 == 0:
                        logger.info(f"Parsed {completed}/{len(file_list)} files")

                except Exception as e:
                    logger.error(f"Error parsing {rel_path}: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Parallel parsing complete: {len(file_list)} files in {elapsed:.2f}s "
            f"({len(file_list)/elapsed:.1f} files/sec)"
        )

        return G, dict(defines), references

    def _parse_single_file(self, rel_path: str) -> Dict[str, Any]:
        """
        Parse a single file and extract nodes/edges.

        Returns:
            Dictionary with:
            - file_node: File node dict
            - code_nodes: List of class/function node dicts
            - defines: Dict of defined identifiers
            - references: List of reference dicts
        """
        file_path = os.path.join(self.repo_path, rel_path)

        # Check if file is readable
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {rel_path}")
            return None

        if not self.repo_map.parse_helper.is_text_file(file_path):
            return None

        try:
            # Read file content
            file_text = self.repo_map.io.read_text(file_path) or ""

            # Create file node
            file_node = {
                'name': rel_path,
                'type': 'FILE',
                'file': rel_path,
                'text': file_text,
                'line': 0,
                'end_line': 0,
                'display_name': rel_path.split('/')[-1]
            }

            # Extract tags using tree-sitter
            tags = list(self.repo_map.get_tags(file_path, rel_path))

            # Process tags into nodes and references
            code_nodes = []
            defines = defaultdict(set)
            references = []

            current_class = None
            current_method = None

            for tag in tags:
                if tag.kind == 'def':
                    # Definition node (class, function, method)
                    node_type, node_name, node_dict = self._create_node_from_tag(
                        tag, rel_path, current_class
                    )

                    if node_dict:
                        code_nodes.append(node_dict)
                        defines[tag.name].add(node_name)

                        # Update current context
                        if tag.type == 'class':
                            current_class = tag.name
                            current_method = None
                        elif tag.type in ['method', 'function']:
                            current_method = tag.name

                elif tag.kind == 'ref':
                    # Reference
                    ref_dict = self._create_reference_from_tag(
                        tag, rel_path, current_class, current_method
                    )
                    if ref_dict:
                        references.append(ref_dict)

            return {
                'file_node': file_node,
                'code_nodes': code_nodes,
                'defines': dict(defines),
                'references': references,
                'rel_path': rel_path
            }

        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            return None

    def _create_node_from_tag(
        self,
        tag,
        rel_path: str,
        current_class: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Create node dictionary from tree-sitter tag.

        Returns:
            (node_type, node_name, node_dict)
        """
        if tag.type == 'class':
            node_type = 'CLASS'
        elif tag.type == 'interface':
            node_type = 'INTERFACE'
        elif tag.type in ['method', 'function']:
            node_type = 'FUNCTION'
        else:
            return None, None, None

        # Create fully qualified node name
        if current_class:
            node_name = f"{rel_path}:{current_class}.{tag.name}"
        else:
            node_name = f"{rel_path}:{tag.name}"

        node_dict = {
            'name': node_name,
            'type': node_type,
            'file': rel_path,
            'line': tag.line,
            'end_line': tag.end_line,
            'display_name': tag.name,
            'class_name': current_class
        }

        return node_type, node_name, node_dict

    def _create_reference_from_tag(
        self,
        tag,
        rel_path: str,
        current_class: str,
        current_method: str
    ) -> Dict[str, Any]:
        """
        Create reference dictionary from tree-sitter tag.
        """
        # Determine source node
        if current_class and current_method:
            source = f"{rel_path}:{current_class}.{current_method}"
        elif current_method:
            source = f"{rel_path}:{current_method}"
        else:
            source = rel_path

        return {
            'source': source,
            'target_ident': tag.name,
            'line': tag.line,
            'end_line': tag.end_line,
            'source_class': current_class,
            'source_method': current_method
        }

    def _merge_file_result(
        self,
        G: nx.MultiDiGraph,
        file_result: Dict[str, Any],
        defines: Dict[str, Set[str]],
        references: List[Dict[str, Any]],
        seen_relationships: Set[Tuple[str, str, str]]
    ):
        """
        Merge single file's parsing results into main graph.
        Thread-safe when called within lock.
        """
        # Add file node
        file_node = file_result['file_node']
        G.add_node(file_node['name'], **file_node)

        # Add code nodes (classes, functions)
        for code_node in file_result['code_nodes']:
            G.add_node(code_node['name'], **code_node)

            # Add CONTAINS edge from file to code node
            rel_key = (file_node['name'], code_node['name'], 'CONTAINS')
            if rel_key not in seen_relationships:
                G.add_edge(
                    file_node['name'],
                    code_node['name'],
                    type='CONTAINS',
                    ident=code_node.get('display_name') or code_node.get('class_name') or code_node['name'].split(':')[-1]
                )
                seen_relationships.add(rel_key)

        # Merge defines
        for ident, node_set in file_result['defines'].items():
            defines[ident].update(node_set)

        # Collect references for later resolution
        references.extend(file_result['references'])
