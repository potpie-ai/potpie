import os
import logging
import math
from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DirectoryWorkUnit:
    """Represents a directory to be parsed as a single task"""
    path: str  # Relative path from repo root
    file_count: int
    files: List[str]  # List of file paths to parse
    depth: int


class DirectoryScannerService:
    """
    Scans repository and divides into parallel work units.
    """

    # File extensions to include - aligned with parsing_helper.py
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.tsx', '.jsx',
        '.c', '.cs', '.cpp', '.cxx', '.cc', '.h', '.hpp',
        '.el', '.ex', '.exs', '.elm',
        '.go', '.java', '.ml', '.mli', '.php', '.ql', '.rb', '.rs',
        '.md', '.mdx', '.txt',
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        '.xml', '.html', '.htm',
        '.css', '.scss', '.sass', '.less',
        '.sh', '.bash', '.zsh', '.fish',
        '.ps1', '.psm1', '.psd1', '.ps1xml',  # PowerShell files
        '.bat', '.cmd',
        '.xsq', '.proto', '.sql',
        '.r', '.R', '.scala', '.kt', '.swift', '.m',
        '.vue', '.svelte',
        '.xaml', '.resx', '.xsd', '.csproj'  # Additional .NET/XML files
    }

    # Directories to skip
    SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.vscode',
                 '.idea', 'bin', 'obj', '.vs', 'packages'}

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.total_files = 0

        # Load configuration from environment at instance creation time
        self.MAX_FILES_PER_TASK = int(os.getenv('MAX_FILES_PER_WORK_UNIT', '2000'))
        self.TARGET_FILES_PER_TASK = int(os.getenv('TARGET_FILES_PER_WORK_UNIT', '1750'))
        self.MIN_FILES_PER_TASK = int(os.getenv('MIN_FILES_PER_WORK_UNIT', '100'))

    def scan_and_divide(self) -> List[DirectoryWorkUnit]:
        """
        Scan repository and create balanced work units.

        Uses a three-phase approach:
        1. Collect all files grouped by directory
        2. Calculate optimal distribution parameters
        3. Create balanced work units using bin-packing

        Returns:
            List of DirectoryWorkUnit objects, each representing a task
        """
        logger.info(f"Scanning repository: {self.repo_path}")
        logger.info(
            f"Configuration: MAX={self.MAX_FILES_PER_TASK}, "
            f"TARGET={self.TARGET_FILES_PER_TASK}, "
            f"MIN={self.MIN_FILES_PER_TASK}"
        )

        # Phase 1: Collect all files grouped by directory
        directory_files = self._collect_all_files()

        if self.total_files == 0:
            logger.warning("No parseable files found in repository")
            return []

        # Phase 2: Calculate optimal distribution
        target_size, expected_count = self._calculate_optimal_distribution(
            self.total_files
        )

        # Phase 3: Create balanced work units
        work_units = self._create_balanced_work_units(
            directory_files,
            target_size
        )

        # Log distribution statistics
        self._log_distribution_stats(work_units)

        logger.info(
            f"Created {len(work_units)} work units for {self.total_files} files"
        )
        return work_units

    def _is_parseable_file(self, filename: str) -> bool:
        """Check if file should be parsed based on extension"""
        _, ext = os.path.splitext(filename.lower())
        return ext in self.SUPPORTED_EXTENSIONS

    def _collect_all_files(self) -> Dict[str, List[str]]:
        """
        Phase 1: Collect all parseable files and group by directory.

        Returns:
            Dict mapping directory path -> list of file paths in that directory
            (files are stored with full relative paths from repo root)
        """
        directory_files = defaultdict(list)

        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]

            # Skip hidden directories
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue

            # Get relative directory path
            rel_dir = os.path.relpath(root, self.repo_path)
            if rel_dir == '.':
                rel_dir = ''

            # Collect parseable files in this directory
            for filename in files:
                if self._is_parseable_file(filename):
                    file_path = os.path.join(root, filename)
                    rel_file_path = os.path.relpath(file_path, self.repo_path)
                    directory_files[rel_dir].append(rel_file_path)
                    self.total_files += 1

        return directory_files

    def _calculate_optimal_distribution(self, total_files: int) -> tuple[int, int]:
        """
        Phase 2: Calculate optimal work unit size for even distribution.

        Args:
            total_files: Total number of files to distribute

        Returns:
            Tuple of (target_files_per_unit, expected_unit_count)
        """
        if total_files == 0:
            return (0, 0)

        if total_files <= self.MIN_FILES_PER_TASK:
            # Very small repo, single work unit
            return (total_files, 1)

        # Calculate expected number of work units based on target
        expected_units = math.ceil(total_files / self.TARGET_FILES_PER_TASK)

        # Calculate optimal size to evenly distribute files
        optimal_size = math.ceil(total_files / expected_units)

        # Ensure optimal size doesn't exceed max
        if optimal_size > self.MAX_FILES_PER_TASK:
            # Recalculate with MAX as constraint
            expected_units = math.ceil(total_files / self.MAX_FILES_PER_TASK)
            optimal_size = math.ceil(total_files / expected_units)

        logger.info(
            f"Distribution plan: {total_files} files → {expected_units} units "
            f"of ~{optimal_size} files each (target: {self.TARGET_FILES_PER_TASK}, "
            f"max: {self.MAX_FILES_PER_TASK})"
        )

        return (optimal_size, expected_units)

    def _create_balanced_work_units(
        self,
        directory_files: Dict[str, List[str]],
        target_size: int
    ) -> List[DirectoryWorkUnit]:
        """
        Phase 3: Create balanced work units using bin-packing algorithm.

        Args:
            directory_files: Dict mapping directory path -> list of files
            target_size: Target number of files per work unit

        Returns:
            List of balanced DirectoryWorkUnit objects
        """
        if not directory_files or target_size == 0:
            return []

        # Data structure to track work units being built
        # Each item: {'files': [], 'directories': set(), 'count': 0}
        work_unit_bins = []

        # Sort directories by file count (largest first) for First-Fit-Decreasing
        sorted_dirs = sorted(
            directory_files.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        logger.info(f"Distributing {len(sorted_dirs)} directories into work units")

        for dir_path, files in sorted_dirs:
            file_count = len(files)

            if file_count == 0:
                continue

            # Try to find a bin that can accommodate this directory
            placed = False

            # Check if directory is too large and needs to be split
            if file_count > self.MAX_FILES_PER_TASK:
                # Split large directory into chunks
                logger.info(
                    f"Splitting large directory {dir_path or 'root'} "
                    f"({file_count} files) into chunks"
                )
                self._chunk_large_directory(
                    dir_path,
                    files,
                    work_unit_bins,
                    target_size
                )
                placed = True
            else:
                # Try to fit entire directory into an existing bin
                for bin_data in work_unit_bins:
                    if bin_data['count'] + file_count <= self.MAX_FILES_PER_TASK:
                        # Check if adding this would be reasonable
                        # (don't overfill beyond target unless necessary)
                        if bin_data['count'] + file_count <= target_size * 1.15:
                            # Add to this bin
                            bin_data['files'].extend(files)
                            bin_data['directories'].add(dir_path)
                            bin_data['count'] += file_count
                            placed = True
                            break

            # If not placed, create new bin
            if not placed:
                work_unit_bins.append({
                    'files': files.copy(),
                    'directories': {dir_path},
                    'count': file_count
                })

        # Convert bins to DirectoryWorkUnit objects
        work_units = []
        for i, bin_data in enumerate(work_unit_bins):
            if bin_data['count'] == 0:
                continue

            # Determine path representation
            dirs = bin_data['directories']
            if len(dirs) == 1:
                path = list(dirs)[0]
            else:
                # Multiple directories - use common prefix or "mixed"
                path = self._get_common_prefix(list(dirs)) or 'mixed'

            work_unit = DirectoryWorkUnit(
                path=path,
                file_count=bin_data['count'],
                files=bin_data['files'],
                depth=0  # Not used in new algorithm
            )
            work_units.append(work_unit)

            logger.info(
                f"Work unit {i+1}: {work_unit.path or 'root'} "
                f"({work_unit.file_count} files from {len(dirs)} dir(s))"
            )

        return work_units

    def _chunk_large_directory(
        self,
        dir_path: str,
        files: List[str],
        work_unit_bins: List[Dict[str, Any]],
        target_size: int
    ):
        """
        Split a large directory into multiple chunks and distribute them.

        Args:
            dir_path: Directory path
            files: List of files in this directory
            work_unit_bins: List of work unit bins being built
            target_size: Target files per work unit
        """
        # Split files into chunks of MAX size
        chunk_size = self.MAX_FILES_PER_TASK
        num_chunks = math.ceil(len(files) / chunk_size)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(files))
            chunk_files = files[start_idx:end_idx]

            # Try to add chunk to existing bin with space
            placed = False
            for bin_data in work_unit_bins:
                if bin_data['count'] + len(chunk_files) <= self.MAX_FILES_PER_TASK:
                    bin_data['files'].extend(chunk_files)
                    bin_data['directories'].add(f"{dir_path} (chunk {i+1}/{num_chunks})")
                    bin_data['count'] += len(chunk_files)
                    placed = True
                    break

            # Create new bin if not placed
            if not placed:
                work_unit_bins.append({
                    'files': chunk_files,
                    'directories': {f"{dir_path} (chunk {i+1}/{num_chunks})"},
                    'count': len(chunk_files)
                })

    def _get_common_prefix(self, paths: List[str]) -> str:
        """
        Get common directory prefix from a list of paths.

        Args:
            paths: List of directory paths

        Returns:
            Common prefix path, or empty string if no common prefix
        """
        if not paths:
            return ''

        if len(paths) == 1:
            return paths[0]

        # Filter out empty strings
        non_empty = [p for p in paths if p]
        if not non_empty:
            return ''

        # Split paths into components
        split_paths = [p.split(os.sep) for p in non_empty]

        # Find common prefix
        common = []
        for components in zip(*split_paths):
            if len(set(components)) == 1:
                common.append(components[0])
            else:
                break

        return os.sep.join(common) if common else ''

    def _log_distribution_stats(self, work_units: List[DirectoryWorkUnit]):
        """
        Log statistics about work unit distribution.

        Args:
            work_units: List of work units to analyze
        """
        if not work_units:
            return

        sizes = [wu.file_count for wu in work_units]
        min_size = min(sizes)
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)
        median_size = sorted(sizes)[len(sizes) // 2]

        logger.info("=" * 60)
        logger.info("Work Unit Distribution Statistics:")
        logger.info(f"  Total work units: {len(work_units)}")
        logger.info(f"  Min size: {min_size} files")
        logger.info(f"  Max size: {max_size} files")
        logger.info(f"  Average size: {avg_size:.1f} files")
        logger.info(f"  Median size: {median_size} files")
        logger.info(f"  Target range: {self.TARGET_FILES_PER_TASK}-{self.MAX_FILES_PER_TASK} files")

        # Count units in different size ranges
        in_target = sum(1 for s in sizes if self.TARGET_FILES_PER_TASK * 0.85 <= s <= self.MAX_FILES_PER_TASK)
        too_small = sum(1 for s in sizes if s < self.MIN_FILES_PER_TASK)
        logger.info(f"  Units in target range: {in_target}/{len(work_units)} ({in_target*100/len(work_units):.1f}%)")
        if too_small > 0:
            logger.warning(f"  Units below minimum ({self.MIN_FILES_PER_TASK}): {too_small}")

        logger.info("=" * 60)
