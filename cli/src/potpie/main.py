import itertools
import click
import logging
import signal
import sys
import os
import time
import requests
from tabulate import tabulate
from dotenv import load_dotenv, find_dotenv

from potpie.utility import Utility
from potpie.server_manager import ServerManager
from potpie.api_wrapper import ApiWrapper

load_dotenv(find_dotenv(), override=True)


server_manager = ServerManager()
api_wrapper = ApiWrapper()


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
        click.secho("Poitre server started successfully.", fg="green", bold=True)
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

    try:
        project_id = api_wrapper.parse_project(repo, branch)

        max_attempts: int = 30
        attempt: int = 0

        while attempt < max_attempts:
            status = api_wrapper.parse_status(project_id)
            logging.info(f"Current status: {status}")

            if status in ["ready", "error"]:

                if status == "ready":
                    click.secho("Parsing completed", fg="green", bold=True)
                else:
                    click.secho("Parsing failed", fg="red", bold=True)
                break

            loading_animation("Parsing in progress")
            attempt += 1

        if attempt >= max_attempts:
            logging.warning("Parsing took too long...")
            click.secho("Tips to resolve this...", fg="cyan", bold=True)
            click.secho(
                "This can be happened due to large repository size, so wait for some time.",
                fg="yellow",
            )
            click.secho(
                "Then manually check the parsing status using 'potpie projects'",
                fg="yellow",
            )

    except Exception as e:
        logging.error(f"Error during parsing: {e}")
        exit(1)


@cli.command()
@click.option("--delete", is_flag=True, help="Enable project deletion mode")
def projects(delete):
    """List all projects and optionally delete selected projects"""

    try:
        projects = api_wrapper.get_list_of_projects()

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
                        status_code = api_wrapper.delete_project(selected_project_id)

                        if status_code == 200:
                            click.echo(
                                f"Project {selected_project['repo_name']} (ID: {selected_project_id}) deleted successfully."
                            )
                        else:
                            click.echo(
                                f"Failed to delete project. Status code: {status_code.status_code}"
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
    base_url: str = Utility.base_url()
    # Sees that user_id is used as the defaultUsername
    user_id = os.getenv("defaultUsername", "defaultuser")
    status = "active"
    project_ids = None
    agent_id = None

    try:
        projects = api_wrapper.get_list_of_projects()

        for idx, project in enumerate(projects, 1):
            click.echo(f"{idx}. {project['repo_name']} (ID: {project['id']})")

        selection = click.prompt(
            "Enter the number of the project to start conversation with", type=int
        )
        if 1 <= selection <= len(projects):
            selected_project = projects[selection - 1]
            selected_project_id: str = selected_project["id"]

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
        agents = api_wrapper.available_agents(system_agent=True)

        for idx, agent in enumerate(agents, 1):
            click.echo(f"{idx}. {agent['name']} (ID: {agent['id']})")

        selection = click.prompt(
            "Enter the number of the agent to start conversation with", type=int
        )
        if 1 <= selection <= len(agents):
            selected_agent = agents[selection - 1]
            selected_agent_id: str = selected_agent["id"]

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
        conversation = api_wrapper.create_conversation(
            title,
            user_id,
            status,
            [project_ids],
            [agent_id],
        )
        click.secho("Conversation created successfully.", fg="green", bold=True)

        print(f"Conversation {conversation}")

    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


@conversation.command()
def list():
    """List all conversations"""
    try:
        conversations = api_wrapper.get_conversation()

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
async def message():
    """Communicate with the agent"""

    conversation_id = None

    try:
        conversations = api_wrapper.get_conversation()

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

            async for message in api_wrapper.interact_with_agent(
                conversation_id, message_details
            ):
                click.echo(message, nl=False)

        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break


if __name__ == "__main__":
    cli()
