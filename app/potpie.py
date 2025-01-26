import itertools
import json
import click
import logging
import signal
import sys
import os
import time
import subprocess
import requests
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ServerManager:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.pid_file = os.path.join(script_dir, "poitre.pid")
        self.celery_log = os.path.join(script_dir, "celery.log")
        self.celery_error_log = os.path.join(script_dir, "celery_error.log")
        self.server_log = os.path.join(script_dir, "server.log")
        self.server_error_log = os.path.join(script_dir, "server_error.log")

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

    def start_docker(self):
        """Start Docker containers using docker-compose"""
        logging.info("Starting Docker containers...")
        try:
            subprocess.Popen(
                ["docker", "compose", "up", "-d"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            while True:
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
                all_running = all(
                    container["State"] == "running" for container in containers
                )

                if all_running and containers:
                    break

                logging.info("Waiting for Docker containers to start...")
                time.sleep(2)

            logging.info("Docker containers started successfully")

        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start Docker containers: {e}")
            raise

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
                return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to check PostgreSQL status: {e}")
            return False

    def run_migrations(self):
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

        except subprocess.CalledProcessError as e:
            if "psycopg2.errors.DuplicateTable" in e.stderr:
                logging.warning("Migration attempted to create a table that already exists. Skipping...")
            else:
                logging.error(f"Failed to run migrations: {e}")
                logging.error(f"Migration error output: {e.stderr}")
                raise
        except FileNotFoundError:
            logging.error(f"Alembic not found in virtual environment: {alembic_path}")
            raise

    def start_server(self):
        """Start the server in a separate process"""
        if os.path.exists(self.pid_file):

            logging.warning("Server is already running")
            return

        if not self.check_environment():
            logging.error("Invalid environment")
            return

        def run_celery():
            """Run Celery worker"""
            with open(self.celery_log, "a+") as stdout, open(self.celery_error_log, "a+") as stderr:
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
                    stdout=stdout,
                    stderr=stderr,
                )

            with open(self.pid_file, "a+") as f:
                f.write(str(celery_process.pid))

            logging.info("Celery worker started...")

        def run_server():
            """Run the server using uvicorn"""


            with open(self.server_log, "a+") as stdout, open(self.server_error_log, "a+") as stderr:


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
                    stdout=stdout,
                    stderr=stderr,
                )

            with open(self.pid_file, "w") as f:
                f.write(str(server_process.pid))
                f.write("\n")

        try:
            self.start_docker()

            if not self.check_postgres():
                raise Exception("Required services are not accessible")

            self.run_migrations()

            self.is_running = True
            logging.info("🚀 Services started:")

            run_server()
            click.secho(
                "Local server is running at http://0.0.0.0:8001",
                fg="green",
                bold=True,
            )
            run_celery()

        except Exception as e:
            logging.error(f"Startup failed: {e}")
            self.stop_server()
            raise

    def stop_server(self):
        """Stop all services in reverse order"""
        if not os.path.exists(self.pid_file):
            logging.warning("Server is not running")
            return
        try:
            logging.info("Stopping Celery worker...")
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
                logging.warning("Server process already terminated {e}")

            logging.info("Stopping Docker containers...")
            try:
                subprocess.run(["docker", "compose", "down"], check=True)
                logging.info("Docker containers stopped")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to stop Docker containers: {e}")

            logging.info("All services stopped successfully")

            os.remove(self.pid_file)

        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signal"""
        logging.info(f"Received shutdown signal ({signum}), stopping server...")
        self.stop_server()
        sys.exit(0)


server_manager = ServerManager()


def loading_animation(message: str):
    """Display loading animation for five seconds"""
    spinner = itertools.cycle(["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
    start_time = time.time()
    while (time.time() - start_time) < 5:
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r\033[K")


@click.group()
def cli():
    """CLI tool for managing the potpie application"""
    pass


@cli.command()
def start():
    """Start the server and all related services"""
    click.secho("Poitre server starting...", fg="blue", bold=True)
    try:
        signal.signal(signal.SIGINT, server_manager.handle_shutdown)
        signal.signal(signal.SIGTERM, server_manager.handle_shutdown)
        server_manager.start_server()
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        sys.exit(1)


@cli.command()
def stop():
    """Stop the server and all related services"""
    server_manager.stop_server()


@cli.command()
@click.argument("repo")
@click.option("--branch", default="main", help="Branch name")
def parse(repo, branch):
    """Parse a Local repository currently"""

    if not os.path.exists(repo):
        click.secho("Invalid repository path", fg="red", bold=True)
        return
    
    repo = os.path.abspath(repo)
    click.secho(f"Starting parsing for repository: {repo}", fg="blue", bold=True)

    repo_details = {"repo_path": repo, "branch_name": branch}

    base_url = "http://localhost:8001"
    response = requests.post(f"{base_url}/api/v1/parse", json=repo_details)

    if response.status_code == 200:
        project_id = response.json()["project_id"]
        print(f"Parsing started for project ID: {project_id}")

        # Poll parsing status
        while True:
            status_response = requests.get(
                f"{base_url}/api/v1/parsing-status/{project_id}"
            )
            status = status_response.json()["status"]
            logging.info(f"Current status: {status}")

            if status in ["ready", "error"]:

                if status == "ready":
                    click.secho("Parsing completed", fg="green", bold=True)
                else:
                    click.secho("Parsing failed", fg="red", bold=True)
                break

            loading_animation("Parsing in progress")
    else:
        print("Failed to start parsing.")


@cli.command()
# TODO: Add the limits which is the limit of the projects to be displayed
@click.option("--delete", is_flag=True, help="Enable project deletion mode")
def projects(delete):
    """List all projects and optionally delete selected projects"""
    base_url = "http://localhost:8001"

    try:
        response = requests.get(f"{base_url}/api/v1/projects/list")
        if response.status_code != 200:
            logging.error("Failed to fetch projects.")
            return

        projects = response.json()

        if not delete:
            # Standard project listing
            table_data = [
                [project["id"], project["repo_name"], project["status"]]
                for project in projects
            ]
            table = tabulate(
                table_data, headers=["ID", "Name", "Status"], tablefmt="fancy_grid"
            )
            click.echo(table)
        else:
            # Simple delete mode
            click.echo("Available Projects:")
            for idx, project in enumerate(projects, 1):
                click.echo(f"{idx}. {project['repo_name']} (ID: {project['id']})")

            try:
                selection = click.prompt(
                    "Enter the number of the project to delete", type=int
                )
                if 1 <= selection <= len(projects):
                    selected_project = projects[selection - 1]
                    selected_project_id = selected_project["id"]

                    confirm = click.confirm(
                        f"Are you sure you want to delete project {selected_project['repo_name']} (ID: {selected_project_id})?"
                    )

                    if confirm:
                        delete_response = requests.delete(
                            f"{base_url}/api/v1/projects",
                            params={"project_id": selected_project_id},
                        )

                        if delete_response.status_code == 200:
                            click.echo(
                                f"Project {selected_project['repo_name']} (ID: {selected_project_id}) deleted successfully."
                            )
                        else:
                            click.echo(
                                f"Failed to delete project. Status code: {delete_response.status_code}"
                            )
                else:
                    click.echo("Invalid project selection.")
            except ValueError:
                click.echo("Invalid input. Please enter a valid project number.")

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


@cli.group()
def conversation():
    """Talk with your conversations"""
    pass


@conversation.command()
@click.argument("title")
def create(title):
    """Create a new conversation"""
    base_url = "http://localhost:8001"

    # Sees that user_id is used as the defaultUsername
    user_id = os.getenv("defaultUsername", "defaultuser")
    status = "active"
    project_ids = None
    agent_id = None

    try:
        response = requests.get(f"{base_url}/api/v1/projects/list")
        if response.status_code != 200:
            logging.error("Failed to fetch projects.")
            return

        projects = response.json()

        for idx, project in enumerate(projects, 1):
            click.echo(f"{idx}. {project['repo_name']} (ID: {project['id']})")

        selection = click.prompt(
            "Enter the number of the project to start conversation with", type=int
        )
        if 1 <= selection <= len(projects):
            selected_project = projects[selection - 1]
            selected_project_id = selected_project["id"]

            confirm = click.confirm(
                f"Are you sure you want to start conversation with {selected_project['repo_name']} (ID: {selected_project_id})?"
            )

            if confirm:
                project_ids = selected_project_id

        else:
            click.echo("Invalid project selection.")
    except ValueError:
        click.echo("Invalid input. Please enter a valid project number.")

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    try:
        response = requests.get(
            f"{base_url}/api/v1/list-available-agents",
            params={"list_system_agents": True},
        )

        if response.status_code != 200:
            logging.error("Failed to fetch agents.")
            return

        agents = response.json()

        for idx, agent in enumerate(agents, 1):
            click.echo(f"{idx}. {agent['name']} (ID: {agent['id']})")

        selection = click.prompt(
            "Enter the number of the agent to start conversation with", type=int
        )
        if 1 <= selection <= len(agents):
            selected_agent = agents[selection - 1]
            selected_agent_id = selected_agent["id"]

            confirm = click.confirm(
                f"Are you sure you want to choose {selected_agent['name']} (ID: {selected_agent_id})?"
            )

            if confirm:
                agent_id = selected_agent_id

        else:
            click.echo("Invalid agent selection.")

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    try:
        response = requests.post(
            f"{base_url}/api/v1/conversations",
            json={
                "user_id": user_id,
                "title": title,
                "status": status,
                "project_ids": [project_ids],
                "agent_ids": [agent_id],
            },
        )
        click.secho("Conversation created successfully.", fg="green", bold=True)

        conversation = response.json()

        print(f"Conversation {conversation}")

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


@conversation.command()
def list():
    """List all conversations"""
    base_url = "http://localhost:8001"

    try:
        response = requests.get(f"{base_url}/api/v1/user/conversations")
        if response.status_code != 200:
            logging.error("Failed to fetch conversations.")
            return

        conversations = response.json()

        table_data = [
            [conversation["id"], conversation["title"], conversation["status"]]
            for conversation in conversations
        ]
        table = tabulate(
            table_data, headers=["ID", "Title", "Status"], tablefmt="fancy_grid"
        )
        click.echo(table)
    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


@conversation.command()
def message():
    """Comunicate with the agent"""

    base_url = "http://localhost:8001"
    conversation_id = None

    try:
        response = requests.get(f"{base_url}/api/v1/user/conversations")
        if response.status_code != 200:
            logging.error("Failed to fetch conversations.")
            return

        conversations = response.json()

        for idx, conversation in enumerate(conversations, 1):
            click.echo(f"{idx}. {conversation['title']} (ID: {conversation['id']})")

        selection = click.prompt(
            "Enter the number of the conversation to start messaging with", type=int
        )
        if 1 <= selection <= len(conversations):
            selected_conversation = conversations[selection - 1]
            selected_conversation_id = selected_conversation["id"]

            confirm = click.confirm(
                f"Are you sure you want to start messaging with {selected_conversation['title']} (ID: {selected_conversation_id})?"
            )

            if confirm:
                conversation_id = selected_conversation_id

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    # Interactive chat loop

    while True:

        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                click.echo("Exiting chat session.")
                break

            message_details = {"content": user_input}
            message_response = requests.post(
                f"{base_url}/api/v1/conversations/{conversation_id}/message/",
                json=message_details,
                stream=True,
            )

            if message_response.status_code == 200:
                for line in message_response.iter_lines():
                    if line:
                        click.echo(f"Agent: {line}")
                else:
                    logging.info("Failed to send message.")

        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break


if __name__ == "__main__":
    cli()
