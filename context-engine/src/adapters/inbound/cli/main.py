import importlib.metadata
import os

import typer

from adapters.outbound.settings_env import EnvContextEngineSettings

try:
    __version__ = importlib.metadata.version("context-engine")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

app = typer.Typer(
    name="context-engine",
    help="Context graph CLI (configure via env; use HTTP API for full sync).",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _cli(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


@app.command("doctor")
def doctor() -> None:
    """Print whether context graph env looks configured."""
    s = EnvContextEngineSettings()
    typer.echo(f"CONTEXT_GRAPH_ENABLED: {s.is_enabled()}")
    typer.echo(f"NEO4J_URI set: {bool(s.neo4j_uri())}")
    typer.echo(f"DATABASE_URL set: {bool(os.getenv('DATABASE_URL') or os.getenv('POSTGRES_URL'))}")
    typer.echo(f"GITHUB_TOKEN set: {bool(os.getenv('CONTEXT_ENGINE_GITHUB_TOKEN') or os.getenv('GITHUB_TOKEN'))}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
