"""Unit tests for Celery parsing tasks."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.celery.tasks.parsing_tasks import process_colgrep_index


pytestmark = pytest.mark.unit


def test_process_colgrep_index_runs_builder():
    """ColGREP task delegates to the sync index builder with a Path base dir."""
    with patch("app.celery.tasks.parsing_tasks.build_colgrep_index") as mock_build:
        process_colgrep_index.run("/tmp/worktree", "/tmp/repos")

    mock_build.assert_called_once_with("/tmp/worktree", Path("/tmp/repos"))
