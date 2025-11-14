import os
import logging
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
        self.MAX_FILES_PER_TASK = int(os.getenv('MAX_FILES_PER_WORK_UNIT', '5000'))
        self.TARGET_FILES_PER_TASK = int(os.getenv('MAX_FILES_PER_WORK_UNIT', '3000'))
        self.MAX_DEPTH = 10

    def scan_and_divide(self) -> List[DirectoryWorkUnit]:
        """
        Scan repository and create work units.

        Returns:
            List of DirectoryWorkUnit objects, each representing a task
        """
        logger.info(f"Scanning repository: {self.repo_path}")
        logger.info(f"Configuration: MAX_FILES_PER_TASK={self.MAX_FILES_PER_TASK}, TARGET_FILES_PER_TASK={self.TARGET_FILES_PER_TASK}")

        # First pass: count files per directory
        dir_file_counts = self._count_files_per_directory()

        # Second pass: create work units using divide-and-conquer
        work_units = self._create_work_units(dir_file_counts)

        logger.info(
            f"Created {len(work_units)} work units for {self.total_files} files"
        )
        return work_units

    def _count_files_per_directory(self) -> Dict[str, int]:
        """
        Walk repository and count parseable files per directory.

        Returns:
            Dict mapping directory path -> cumulative file count (includes subdirectories)
        """
        dir_counts = defaultdict(int)

        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]

            # Skip hidden directories
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue

            rel_root = os.path.relpath(root, self.repo_path)
            if rel_root == '.':
                rel_root = ''

            # Count parseable files in this directory
            file_count = sum(
                1 for f in files
                if self._is_parseable_file(f)
            )

            self.total_files += file_count

            # Accumulate counts for this directory and all parents
            current = rel_root
            while True:
                dir_counts[current] += file_count
                if not current:
                    break
                current = os.path.dirname(current)

        return dir_counts

    def _is_parseable_file(self, filename: str) -> bool:
        """Check if file should be parsed based on extension"""
        _, ext = os.path.splitext(filename.lower())
        return ext in self.SUPPORTED_EXTENSIONS

    def _create_work_units(
        self,
        dir_counts: Dict[str, int]
    ) -> List[DirectoryWorkUnit]:
        """
        Create work units using greedy directory splitting.

        Strategy:
        1. Start with root directory
        2. If directory has > MAX_FILES_PER_TASK, split into subdirectories
        3. Recursively process subdirectories
        4. Create work unit when directory is small enough
        """
        work_units = []

        # Process root
        self._split_directory(
            path='',
            dir_counts=dir_counts,
            depth=0,
            work_units=work_units
        )

        return work_units

    def _split_directory(
        self,
        path: str,
        dir_counts: Dict[str, int],
        depth: int,
        work_units: List[DirectoryWorkUnit]
    ):
        """
        Recursively split directory into work units.
        """
        full_path = os.path.join(self.repo_path, path) if path else self.repo_path
        file_count = dir_counts.get(path, 0)

        # Base cases
        if file_count == 0:
            return

        if depth > self.MAX_DEPTH:
            # Max depth reached, create work unit even if large
            logger.warning(
                f"Max depth reached for {path} with {file_count} files"
            )
            self._create_work_unit(path, depth, work_units)
            return

        if file_count <= self.MAX_FILES_PER_TASK:
            # Small enough, create work unit
            self._create_work_unit(path, depth, work_units)
            return

        # Directory too large, split into subdirectories
        logger.info(f"Splitting {path} ({file_count} files) into subdirectories")

        try:
            subdirs = [
                d for d in os.listdir(full_path)
                if os.path.isdir(os.path.join(full_path, d))
                and d not in self.SKIP_DIRS
                and not d.startswith('.')
            ]
        except OSError as e:
            logger.error(f"Error listing directory {path}: {e}")
            return

        if not subdirs:
            # No subdirectories, must create work unit even if large
            logger.warning(
                f"No subdirectories in {path} with {file_count} files"
            )
            self._create_work_unit(path, depth, work_units)
            return

        # Recursively process subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir) if path else subdir
            self._split_directory(
                path=subdir_path,
                dir_counts=dir_counts,
                depth=depth + 1,
                work_units=work_units
            )

    def _create_work_unit(
        self,
        path: str,
        depth: int,
        work_units: List[DirectoryWorkUnit]
    ):
        """
        Create a work unit for a directory by collecting all parseable files.
        """
        full_path = os.path.join(self.repo_path, path) if path else self.repo_path
        files = []

        # Walk directory tree and collect all files
        for root, dirs, filenames in os.walk(full_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]

            # Skip hidden directories
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue

            for filename in filenames:
                if self._is_parseable_file(filename):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    files.append(rel_path)

        if files:
            work_unit = DirectoryWorkUnit(
                path=path,
                file_count=len(files),
                files=files,
                depth=depth
            )
            work_units.append(work_unit)
            logger.info(
                f"Created work unit: {path or 'root'} ({len(files)} files, depth {depth})"
            )
