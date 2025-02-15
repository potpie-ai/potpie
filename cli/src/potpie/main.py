"""
Potpie CLI Tool

This module provides a command-line interface (CLI) for managing and interacting with
the Potpie application. Potpie allows users to parse repositories, manage projects,
start conversations, and communicate with AI agents.

"""

import asyncio
import os
import sys
import logging
import itertools
import time
import signal

import click
import requests
from tabulate import tabulate
from dotenv import load_dotenv, find_dotenv

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
        logging.error("Error during startup: %s", e)


@cli.command()
def stop():
    """Stop the server and all related services"""
    try:
        server_manager.stop_server()
    except Exception as e:
        logging.error("Error during shutdown: %s", e)


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
                        f"Delete {selected_project['repo_name']} with (ID: {selected_project_id})?"
                    )

                    if confirm:
                        status_code = api_wrapper.delete_project(selected_project_id)

                        if status_code == 200:
                            click.echo(f"Project {selected_project['repo_name']}")
                            click.echo(
                                f"ID: {selected_project_id}) deleted successfully."
                            )
                        else:
                            click.echo(
                                f"Failed to delete project. Status code: {status_code}"
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


def handle_api_error(func):
    """Decorator for handling API errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
    return wrapper

@conversation.command()
@click.argument("title")
@click.option("--max-length", default=100, help="Maximum title length")
@handle_api_error
def create(title, max_length):
    """Create a new conversation"""
    if not title.strip():
        click.secho("Title cannot be empty", fg="red")
        return
    if len(title) > max_length:
        click.secho(f"Title exceeds maximum length of {max_length} characters", fg="red")
        return

    # Sees that user_id is used as the defaultUsername
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
                f"Wanna start conversation with {selected_project['repo_name']} ?"
            )

            if confirm:
                project_ids = selected_project_id

        else:
            click.echo("Invalid project selection.")
    except ValueError:
        click.echo("Invalid input. Please enter a valid project number.")
    except requests.RequestException as e:
        logging.error("Network error occurred: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)

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
                f"Wanna choose this {selected_agent['name']} agent?"
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
            title=title,
            project_id_list=[project_ids],
            agent_id_list=[agent_id],
        )
        click.secho("Conversation created successfully.", fg="green", bold=True)
        print(f"Conversation {conversation}")
    except requests.RequestException as e:
        logging.error(f"Network error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

@conversation.command(name="list")
def list_conversations():
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
        logging.error("Network error occurred: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)


@conversation.command()
def message():
    """Communicate with the agent"""
    asyncio.run(_message())


async def _message():
    """Actual async function for message handling"""
    conversation_id: str = ""

    try:
        conversations: dict = api_wrapper.get_conversation()

        for idx, conversation in enumerate(conversations, 1):
            click.echo(
                f"{idx}. {conversation.get('title')} (ID: {conversation.get('id')})"
            )

        selection = click.prompt(
            "Enter the number of the conversation to start messaging with", type=int
        )
        if 1 <= selection <= len(conversations):
            selected_conversation = conversations[selection - 1]
            selected_conversation_id: str = selected_conversation["id"]

            confirm = click.confirm(
                f"Wanna start messaging with {selected_conversation['title']}?"
            )

            if confirm:
                conversation_id = selected_conversation_id

    except requests.RequestException as e:
        logging.error("Network error occurred: %s", e)

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)

    # Interactive chat loop

    while True:

        try:
            user_input: str = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                click.echo("Exiting chat session.")
                break

            async for message in api_wrapper.interact_with_agent(
                conversation_id=conversation_id, content=user_input
            ):
                click.echo(message, nl=False)

        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break


if __name__ == "__main__":
    cli()
