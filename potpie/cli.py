"""Potpie CLI entry point."""

import asyncio
import sys
from pathlib import Path

import click

from potpie import __version__


@click.group()
@click.version_option(version=__version__, prog_name="potpie")
def cli():
    """Potpie - AI-powered code analysis.

    Analyze codebases and ask questions about code using LLMs.
    """
    pass


@cli.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="LLM model to use (e.g., gpt-4o)")
def index(path: str, model: str | None):
    """Index a repository at PATH.

    Parses source files and builds a knowledge graph for querying.

    Examples:

        potpie index .

        potpie index ~/projects/myrepo
    """
    from potpie import Potpie, PotpieConfig

    async def run():
        config_kwargs = {"project_path": path}
        if model:
            config_kwargs["model"] = model

        pp = Potpie(PotpieConfig(**config_kwargs))
        try:
            click.echo(f"Indexing {Path(path).resolve()}...")
            result = await pp.index()
            click.echo(
                f"Indexed {result.file_count} files, "
                f"{len(result.nodes)} nodes, "
                f"{len(result.edges)} edges"
            )
            if result.error_count > 0:
                click.echo(f"  ({result.error_count} files had errors)", err=True)
        finally:
            await pp.close()

    try:
        asyncio.run(run())
    except NotImplementedError as e:
        click.echo(f"Not yet implemented: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option(
    "--path", "-p", default=".", type=click.Path(exists=True), help="Repository path"
)
@click.option("--model", "-m", default=None, help="LLM model to use")
def ask(question: str, path: str, model: str | None):
    """Ask a question about a codebase.

    Examples:

        potpie ask "Where is authentication implemented?"

        potpie ask "How does the caching layer work?" --path ~/myrepo
    """
    from potpie import Potpie, PotpieConfig

    async def run():
        config_kwargs = {"project_path": path}
        if model:
            config_kwargs["model"] = model

        pp = Potpie(PotpieConfig(**config_kwargs))
        try:
            response = await pp.ask(question)
            click.echo(response.content)
            if response.sources:
                click.echo("\nSources:")
                for source in response.sources:
                    click.echo(f"  - {source}")
        finally:
            await pp.close()

    try:
        asyncio.run(run())
    except NotImplementedError as e:
        click.echo(f"Not yet implemented: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--path", "-p", default=".", type=click.Path(exists=True), help="Repository path"
)
@click.option("--model", "-m", default=None, help="LLM model to use")
def chat(path: str, model: str | None):
    """Start an interactive chat session about a codebase.

    Examples:

        potpie chat

        potpie chat --path ~/myrepo --model gpt-4o
    """
    from potpie import Potpie, PotpieConfig

    async def run():
        config_kwargs = {"project_path": path}
        if model:
            config_kwargs["model"] = model

        pp = Potpie(PotpieConfig(**config_kwargs))
        conversation_id = None

        click.echo(f"Potpie v{__version__} - Interactive Code Chat")
        click.echo(f"Repository: {Path(path).resolve()}")
        click.echo("Type 'quit' or 'exit' to end the session.\n")

        try:
            while True:
                try:
                    user_input = click.prompt("You", prompt_suffix="> ")
                except click.Abort:
                    break

                if user_input.lower() in ("quit", "exit", "q"):
                    break

                if not user_input.strip():
                    continue

                try:
                    response = await pp.chat(user_input, conversation_id)
                    click.echo(f"\nAssistant: {response.content}\n")
                except NotImplementedError as e:
                    click.echo(f"\nNot yet implemented: {e}\n", err=True)
                    break

        finally:
            await pp.close()
            click.echo("\nGoodbye!")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8080, help="Port to bind to")
@click.option(
    "--path", default=".", type=click.Path(exists=True), help="Repository path"
)
def serve(host: str, port: int, path: str):
    """Start an HTTP API server.

    Exposes Potpie functionality via a REST API.

    Examples:

        potpie serve

        potpie serve --host 0.0.0.0 --port 9000
    """
    click.echo(f"Starting Potpie API server on {host}:{port}...")
    click.echo(f"Repository: {Path(path).resolve()}")
    click.echo("Not yet implemented", err=True)
    sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
