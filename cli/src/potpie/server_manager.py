import time
import json
import logging
import os
import signal
import subprocess
import sys
from dotenv import load_dotenv, find_dotenv
from potpie.utility import Utility

load_dotenv(find_dotenv(), override=True)


class ServerManagerException(Exception):
    """Base class for exceptions in this module."""
    pass


class StartServerError(ServerManagerException):
    def __init__(self, message="Starting the server has an error"):
        self.message = message
        super().__init__(self.message)
    
class StopServerError(ServerManagerException):
    def __init__(self, message="Stopping the server has an error"):
        self.message = message
        super().__init__(self.message)


class EnvironmentError(ServerManagerException):
    """Exception raised for errors in the environment."""

    def __init__(self, message="Invalid environment"):
        self.message = message
        super().__init__(self.message)


class DockerError(ServerManagerException):
    """Exception raised for errors related to Docker."""

    def __init__(self, message="Docker error"):
        self.message = message
        super().__init__(self.message)


class PostgresError(ServerManagerException):
    """Exception raised for errors related to PostgreSQL."""

    def __init__(self, message="PostgreSQL error"):
        self.message = message
        super().__init__(self.message)


class MigrationError(ServerManagerException):
    """Exception raised for errors during database migrations."""

    def __init__(self, message="Migration error"):
        self.message = message
        super().__init__(self.message)


class ServerManager:
    def __init__(self):

        utility: Utility = Utility()
        self.pid_file = Utility.create_path_of_pid_file()
        self.server_log = utility.server_log_file
        self.celery_log = utility.celery_log_file

    def check_environment(self) -> bool:
        """Check if we're in the development environment"""
        env = os.getenv("ENV")
        logging.info(f"Current ENV value: {env}")
        if env == "development":
            return True
        else:
            logging.warning(
                f"Invalid environment: {env}. This command is only available in the development environment."
            )
            return False

    def is_docker_installed(self) -> bool:
        """Check if Docker is installed"""
        logging.info("Checking if Docker is installed...")
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logging.info("Docker is installed")
                return True
            else:
                logging.error(
                    f"Docker check failed: {result.stderr.strip() or result.stdout.strip()}"
                )
                return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to check Docker status: {e}")
            return False

    def start_docker(self):
        """Start Docker containers using docker-compose"""
        logging.info("Starting Docker containers...")

        is_docker_installed = self.is_docker_installed()

        if not is_docker_installed:
            logging.error("Docker is not installed. Aborting...")
            raise DockerError("Docker is not installed")

        try:
            subprocess.Popen(
                ["docker", "compose", "up", "-d"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            max_attempts: int = 30
            attempt: int = 0
            while attempt < max_attempts:
                output = subprocess.run(
                    ["docker", "compose", "ps", "--format", "json"],
                    capture_output=True,
                    text=True,
                )

                containers = [
                    json.loads(line)
                    for line in output.stdout.splitlines()
                    if line.strip()
                ]
                all_running: bool = all(
                    container["State"] == "running" for container in containers
                )

                if all_running and containers:
                    break

                logging.info("Waiting for Docker containers to start...")
                time.sleep(2)
                attempt += 1

            if attempt >= max_attempts:
                logging.error(
                    "Docker containers does not run within the time 60 seconds"
                )

                raise DockerError("Docker containers failed to start in time.")

            logging.info("Docker containers started successfully")

        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start Docker containers: {e}")
            raise DockerError("Failed to start Docker containers")

    def check_postgres(self) -> bool:
        """Check if PostgreSQL server is running"""
        logging.info("Checking if PostgreSQL server is running...")
        try:
            result = subprocess.run(
                ["docker", "exec", "potpie_postgres", "pg_isready", "-U", "postgres"],
                capture_output=True,
                text=True,
            )

            if (
                result.returncode == 0
                and "accepting connections" in result.stdout.lower()
            ):
                logging.info("PostgreSQL is running and accepting connections")
                return True
            else:
                logging.error(
                    f"PostgreSQL check failed: {result.stderr.strip() or result.stdout.strip()}"
                )
                raise PostgresError(
                    f"PostgreSQL check failed: {result.stderr.strip() or result.stdout.strip()}"
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to check PostgreSQL status: {e}")
            raise PostgresError("Failed to check PostgreSQL status")

    def run_migrations(self) -> None:
        """Run database migrations using alembic from virtual environment"""
        logging.info("Running database migrations...")
        alembic_path = None
        try:
            venv_path = os.getenv("VIRTUAL_ENV", ".venv")
            if sys.platform == "win32":
                alembic_path = os.path.join(venv_path, "Scripts", "alembic")
            else:
                alembic_path = os.path.join(venv_path, "bin", "alembic")

            result = subprocess.run(
                [alembic_path, "upgrade", "head"],
                check=True,
                capture_output=True,
                text=True,
            )
            logging.info("Database migrations completed successfully")
            logging.debug(f"Migration output: {result.stdout}")

            if result.returncode == 1:
                raise MigrationError("Failed to run the migrations")

        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to run migrations: {e}")
            raise MigrationError(f"Failed to run migrations {e}")

        except FileNotFoundError:
            logging.error(f"Alembic not found in virtual environment: {alembic_path}")
            raise MigrationError(
                f"Alembic not found in virtual environment: {alembic_path}"
            )

    def run_celery(self):
            """Run Celery worker"""
            try:
                with open(self.celery_log, "a+") as logout:
                    celery_queue_name = os.getenv("CELERY_QUEUE_NAME", "dev")
                    celery_process = subprocess.Popen(
                        [
                            "celery",
                            "-A",
                        "app.celery.celery_app",
                        "worker",
                        "--loglevel=info",
                        "-Q",
                        f"{celery_queue_name}_process_repository",
                        "-E",
                        "--concurrency=1",
                        "--pool=solo",
                        ],
                        stdout=logout,
                        stderr=logout,
                    )
                if(celery_process.returncode != 0):
                    raise StartServerError(f" Start Celery Failed: {e}")

                with open(self.pid_file, "a+") as f:
                    f.write(str(celery_process.pid))
            except Exception as e:
                logging.error(f"Failed to start Celery worker: {e}")
                raise StartServerError(f"Failed to start Celery worker: {e}")
            
    def run_server(self):
        """Run the server using uvicorn"""
        try:
            with open(self.server_log, "a+") as logout:

                server_process = subprocess.Popen(
                    [
                        "gunicorn",
                        "--worker-class",
                        "uvicorn.workers.UvicornWorker",
                        "--workers",
                        "1",
                        "--timeout",
                        "1800",
                        "--bind",
                        "0.0.0.0:8001",
                        "--log-level",
                        "debug",
                        "app.main:app",
                    ],
                    stdout=logout,
                    stderr=logout,
                )
            
                if(server_process.returncode != 0):
                    raise StartServerError(f" Start Server Failed: {e}")

                with open(self.pid_file, "w") as f:
                    f.write(str(server_process.pid))
                    f.write("\n")

        except Exception as e:
            StartServerError(f" Start Server Failed: {e}")
            
    
            
    def start_server(self):
        if os.path.exists(self.pid_file):
            logging.warning("Server is already running")
            raise StartServerError("Server is already running")

        if not self.check_environment():
            logging.error("Invalid environment")
            raise StartServerError("Invalid Environment")

        try:
            self.start_docker()

            if not self.check_postgres():
                raise StartServerError("PostgreSQL check failed")

            self.run_migrations()

            self.is_running = True
            logging.info("🚀 Services started:")
            self.run_server()
            logging.info(f"Server is up and running at {Utility.base_url}")
            self.run_celery()
            logging.info("Startup is running")

        except Exception as e:
            logging.error(f"Startup failed: {e}")
            try:
                self.stop_server()
            except StopServerError as stop_error:
                logging.warning(f"StopServerError during rollback: {stop_error}")
            raise StartServerError(f"Startup failed: {e}")  # Ensure correct exception


    
    def stop_server(self):
        """Stop all services in reverse order"""
        if not os.path.exists(self.pid_file):
            logging.warning("Server is not running")
            raise StopServerError("Server is not running")
        try:
            logging.info("Stopping servers...")
            stack = []
            try:
                with open(self.pid_file, "r") as f:
                    pids = [int(pid) for pid in f.read().strip().split()]
                    stack.extend(pids)

                while stack:
                    pid = stack.pop()
                    try:
                        os.kill(pid, signal.SIGTERM)
                        logging.info(f"Terminated process with PID: {pid}")
                    except ProcessLookupError:
                        logging.warning(f"Process with PID {pid} not found")

            except Exception as e:
                logging.warning(f"Server process already terminated {e}")
                raise StopServerError(f"Server process already terminated {e}")

            logging.info("Stopping Docker containers...")
            try:
                subprocess.run(["docker", "compose", "down"], check=True)
                logging.info("Docker containers stopped")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to stop Docker containers: {e}")
                raise StopServerError(f"Failed to stop Docker containers: {e}")

            logging.info("All services stopped successfully")

            os.remove(self.pid_file)

        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            raise StopServerError(f"Error during shutdown: {e}")

    def handle_shutdown(self, signum):
        """Handle shutdown signal,

        Args:
            signum: Signal number

        note: we can also aff the frames

        """
        logging.info(f"Received shutdown signal ({signum}), stopping server...")
        self.stop_server()
        sys.exit(0)
