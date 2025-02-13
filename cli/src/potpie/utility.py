import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv(), override=True)


class Utility:

    def __init__(self):
        if not hasattr(self, "log_file"):
            self._log_file = self._create_logs_files()

    @property
    def log_file(self):
        return self._log_file

    @staticmethod
    def base_url() -> str:
        return "http://localhost:8001"

    @staticmethod
    def get_user_id() -> str:
        return os.getenv("defaultUsername", "defaultuser")

    def _create_logs_files(self) -> str:
        """Create log files for server communications"""

        if os.name == "nt":
            log_dir = os.path.join(Path(os.getenv("LOCALAPPDATA")), "potpie/logs")
        else:
            log_dir = os.path.join(Path.home(), ".local/share/potpie/logs")

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_file = os.path.join(log_dir, "potpie.log")

        if not os.path.exists(log_file):
            with open(log_file, "w") as _:
                pass

        return log_file

    @staticmethod
    def create_path_of_pid_file() -> str:
        """Create a PID file for the server"""
        if os.name == "nt":
            config_dir = os.path.join(Path(os.getenv("LOCALAPPDATA")), "potpie")
        else:
            config_dir = os.path.join(Path.home(), ".config/potpie")

        Path(config_dir).mkdir(parents=True, exist_ok=True)

        pid_file = os.path.join(config_dir, "potpie.pid")
        print(pid_file)
        return pid_file
