"""
Unit tests for DirectoryScannerService balanced work unit distribution.

Tests the bin-packing algorithm that distributes files evenly across work units.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from app.modules.parsing.graph_construction.directory_scanner_service import (
    DirectoryScannerService,
    DirectoryWorkUnit,
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def reset_env_vars(monkeypatch):
    """Reset environment variables to defaults for each test."""
    # Clear any existing values
    monkeypatch.delenv('MAX_FILES_PER_WORK_UNIT', raising=False)
    monkeypatch.delenv('TARGET_FILES_PER_WORK_UNIT', raising=False)
    monkeypatch.delenv('MIN_FILES_PER_WORK_UNIT', raising=False)


def create_files(repo_path: str, structure: dict):
    """
    Helper to create a directory structure with files.

    Args:
        repo_path: Root path of the repository
        structure: Dict mapping directory paths to number of files
                   e.g., {'': 5, 'src': 10, 'src/utils': 3}
    """
    for dir_path, num_files in structure.items():
        full_dir = os.path.join(repo_path, dir_path) if dir_path else repo_path
        os.makedirs(full_dir, exist_ok=True)

        for i in range(num_files):
            file_path = os.path.join(full_dir, f"file_{i}.py")
            Path(file_path).touch()


class TestDirectoryScannerBalancedDistribution:
    """Tests for balanced work unit distribution algorithm."""

    def test_small_repo_single_unit(self, temp_repo):
        """Test that a small repo creates a single work unit."""
        # Create repo with 50 files (below MIN threshold of 100)
        create_files(temp_repo, {'': 50})

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        assert len(work_units) == 1
        assert work_units[0].file_count == 50

    def test_even_distribution_multiple_dirs(self, temp_repo):
        """Test even distribution across multiple small directories."""
        # Create 20 directories with 100 files each = 2000 total
        # With TARGET=1750, MAX=2000, should create ~1-2 work units
        structure = {f'dir_{i}': 100 for i in range(20)}
        create_files(temp_repo, structure)

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        assert total_files == 2000

        # Verify distribution is balanced (all units within reasonable range)
        sizes = [wu.file_count for wu in work_units]
        avg_size = sum(sizes) / len(sizes)

        # All units should be within 20% of average
        for size in sizes:
            assert abs(size - avg_size) / avg_size <= 0.20

    def test_large_directory_chunking(self, temp_repo, monkeypatch):
        """Test that large directories are properly chunked."""
        # Set small MAX for testing
        monkeypatch.setenv('MAX_FILES_PER_WORK_UNIT', '500')
        monkeypatch.setenv('TARGET_FILES_PER_WORK_UNIT', '400')

        # Create one directory with 1500 files
        create_files(temp_repo, {'large_dir': 1500})

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        assert total_files == 1500

        # Should create 3-4 work units (1500 / 500 = 3)
        assert len(work_units) >= 3

        # No unit should exceed MAX
        for wu in work_units:
            assert wu.file_count <= 500

    def test_mixed_directory_sizes(self, temp_repo, monkeypatch):
        """Test distribution with mix of small and medium directories."""
        monkeypatch.setenv('MAX_FILES_PER_WORK_UNIT', '1000')
        monkeypatch.setenv('TARGET_FILES_PER_WORK_UNIT', '850')

        # Create mixed structure:
        # - 10 small dirs (50 files each) = 500 files
        # - 5 medium dirs (200 files each) = 1000 files
        # Total: 1500 files -> should create ~2 work units
        structure = {}
        for i in range(10):
            structure[f'small_{i}'] = 50
        for i in range(5):
            structure[f'medium_{i}'] = 200

        create_files(temp_repo, structure)

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        assert total_files == 1500

        # Should create 2 work units (1500 / 850 ≈ 2)
        assert len(work_units) in [1, 2, 3]

        # All units should be within target range (with some tolerance)
        for wu in work_units:
            assert wu.file_count <= 1000  # MAX
            # Allow some small units for flexibility
            assert wu.file_count >= 100 or len(work_units) == 1

    def test_optimal_distribution_calculation(self, temp_repo):
        """Test the optimal distribution calculation logic."""
        # Create 10,000 files across directories
        structure = {f'dir_{i}': 100 for i in range(100)}
        create_files(temp_repo, structure)

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        assert total_files == 10000

        # With TARGET=1750, MAX=2000, optimal would be:
        # expected_units = ceil(10000 / 1750) = 6
        # optimal_size = ceil(10000 / 6) = 1667
        # Should create 5-7 units

        # Verify we get close to expected distribution
        assert 5 <= len(work_units) <= 8

        # All units should respect MAX limit
        for wu in work_units:
            assert wu.file_count <= 2000

        # Most units should be reasonably sized (not too small)
        large_units = [wu for wu in work_units if wu.file_count >= 1000]
        assert len(large_units) >= len(work_units) * 0.7  # At least 70% should be substantial

    def test_no_files_empty_result(self, temp_repo):
        """Test that empty repo returns empty work units."""
        # Create directories but no parseable files
        os.makedirs(os.path.join(temp_repo, 'empty_dir'))

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        assert len(work_units) == 0

    def test_skips_excluded_directories(self, temp_repo):
        """Test that excluded directories are skipped."""
        # Create files in excluded directories
        create_files(temp_repo, {
            '': 10,
            'node_modules': 100,  # Should be skipped
            '.git': 50,  # Should be skipped
            'src': 20
        })

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        # Should only count files in '' and 'src' (30 total)
        assert total_files == 30

    def test_only_parseable_extensions(self, temp_repo):
        """Test that only files with supported extensions are included."""
        # Create mix of parseable and non-parseable files
        root = temp_repo

        # Parseable files
        Path(os.path.join(root, 'test.py')).touch()
        Path(os.path.join(root, 'app.js')).touch()
        Path(os.path.join(root, 'component.tsx')).touch()

        # Non-parseable files
        Path(os.path.join(root, 'image.png')).touch()
        Path(os.path.join(root, 'data.csv')).touch()
        Path(os.path.join(root, 'readme.doc')).touch()

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        # Should only count 3 parseable files
        assert total_files == 3

    def test_bin_packing_efficiency(self, temp_repo, monkeypatch):
        """Test that bin-packing creates efficient distribution."""
        monkeypatch.setenv('MAX_FILES_PER_WORK_UNIT', '1000')
        monkeypatch.setenv('TARGET_FILES_PER_WORK_UNIT', '850')

        # Create scenario designed to test bin-packing:
        # - 2 dirs with 700 files each = 1400 (should fit in 2 bins)
        # - 4 dirs with 150 files each = 600 (should fill remaining space)
        # Total: 2000 files
        structure = {
            'large_1': 700,
            'large_2': 700,
            'medium_1': 150,
            'medium_2': 150,
            'medium_3': 150,
            'medium_4': 150,
        }
        create_files(temp_repo, structure)

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        total_files = sum(wu.file_count for wu in work_units)
        assert total_files == 2000

        # Should efficiently pack into 2-4 work units
        assert len(work_units) <= 4

        # No unit should exceed MAX
        for wu in work_units:
            assert wu.file_count <= 1000

        # Total should be distributed reasonably (no single unit with majority)
        if len(work_units) > 1:
            max_unit_size = max(wu.file_count for wu in work_units)
            assert max_unit_size <= 1000  # Respects MAX

    def test_respects_max_limit(self, temp_repo, monkeypatch):
        """Test that no work unit exceeds MAX_FILES_PER_WORK_UNIT."""
        monkeypatch.setenv('MAX_FILES_PER_WORK_UNIT', '300')

        # Create various directory sizes
        structure = {
            'huge': 1000,  # Will be chunked
            'medium': 200,
            'small': 50,
        }
        create_files(temp_repo, structure)

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        # Verify no unit exceeds MAX
        for wu in work_units:
            assert wu.file_count <= 300

    def test_work_unit_file_list_completeness(self, temp_repo):
        """Test that all files are included exactly once."""
        structure = {'dir1': 50, 'dir2': 75, 'dir3': 100}
        create_files(temp_repo, structure)

        scanner = DirectoryScannerService(temp_repo)
        work_units = scanner.scan_and_divide()

        # Collect all files from work units
        all_files_in_units = []
        for wu in work_units:
            all_files_in_units.extend(wu.files)

        # Verify total count
        assert len(all_files_in_units) == 225

        # Verify no duplicates
        assert len(all_files_in_units) == len(set(all_files_in_units))

        # Verify all files exist
        for file_path in all_files_in_units:
            full_path = os.path.join(temp_repo, file_path)
            assert os.path.exists(full_path)

    def test_configuration_defaults(self, temp_repo):
        """Test that default configuration values are used when env vars not set."""
        scanner = DirectoryScannerService(temp_repo)

        # Should use defaults when env vars not set
        assert scanner.MAX_FILES_PER_TASK == 2000
        assert scanner.TARGET_FILES_PER_TASK == 1750
        assert scanner.MIN_FILES_PER_TASK == 100

    def test_custom_configuration(self, temp_repo, monkeypatch):
        """Test that custom configuration from env vars is respected."""
        monkeypatch.setenv('MAX_FILES_PER_WORK_UNIT', '5000')
        monkeypatch.setenv('TARGET_FILES_PER_WORK_UNIT', '4000')
        monkeypatch.setenv('MIN_FILES_PER_WORK_UNIT', '200')

        scanner = DirectoryScannerService(temp_repo)

        assert scanner.MAX_FILES_PER_TASK == 5000
        assert scanner.TARGET_FILES_PER_TASK == 4000
        assert scanner.MIN_FILES_PER_TASK == 200
