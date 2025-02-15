"""
Utility Module for Potpie

This module provides utility functions and configurations for the Potpie application.
It includes functionalities for managing log files, retrieving environment variables,
and handling configuration paths.

"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class Utility:
    """A utility class for managing log files and configuration paths."""

    def __init__(self):
        """Initialize Utility class and create log files."""
        if not hasattr(self, "log_file"):
            self._server_log_file = self._create_logs_files("server.log")
            self._celery_log_file = self._create_logs_files("celery.log")

    @property
    def server_log_file(self):
        """Return the server log file path."""
        return self._server_log_file

    @property
    def celery_log_file(self):
        """Return the Celery log file path."""
        return self._celery_log_file

    @staticmethod
    def base_url() -> str:
        """Retrieve the base URL from environment variables."""
        return os.getenv("POTPIE_BASE_URL", "http://localhost:8001")

    @staticmethod
    def get_user_id() -> str:
        """Retrieve the default username from environment variables."""
        return os.getenv("defaultUsername", "defaultuser")

    def _create_logs_files(self, filename) -> str:
        """Create log files for server communications."""
        if os.name == "nt":
            log_dir = os.path.join(Path(os.getenv("LOCALAPPDATA")), "potpie/logs")
        else:
            log_dir = os.path.join(Path.home(), ".local/share/potpie/logs")

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_file = os.path.join(log_dir, filename)

        if not os.path.exists(log_file):
            with open(
                log_file, "w", encoding="utf-8"
            ) as _:  # Explicitly specify encoding
                pass

        return log_file

    @staticmethod
    def create_path_of_pid_file() -> str:
        """Create and return the path for the PID file."""
        if os.name == "nt":
            config_dir = os.path.join(Path(os.getenv("LOCALAPPDATA")), "potpie")
        else:
            config_dir = os.path.join(Path.home(), ".config/potpie")

        Path(config_dir).mkdir(parents=True, exist_ok=True)

        pid_file = os.path.join(config_dir, "potpie.pid")
        print(pid_file)
        return pid_file
