from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="Potpie database seed (JSON profiles).")


@app.command("apply")
def cmd_apply(
    profile: str = typer.Option("default", "--profile", "-p", help="Data profile under seed/data/"),
    email: str = typer.Option(..., "--email", "-e", help="Target user email (must match user.json)"),
    force: bool = typer.Option(False, "--force", "-f", help="Wipe existing seed user + state, then apply"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Load env only; do not write DB"),
) -> None:
    """Insert seed rows for the given profile."""
    from seed.apply import run_apply

    try:
        out = run_apply(profile, email, force=force, dry_run=dry_run)
    except Exception as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(out)


@app.command("wipe")
def cmd_wipe(
    email: str | None = typer.Option(None, "--email", "-e"),
    uid: str | None = typer.Option(None, "--uid", "-u"),
    profile: str | None = typer.Option(
        None, "--profile", "-p", help="Also remove seed/.state/<profile>.json"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm destructive delete"),
) -> None:
    """Delete seeded user data (Potpie + workflows tables) for email or uid."""
    from seed.wipe import run_wipe

    try:
        out = run_wipe(email=email, uid=uid, profile=profile, dry_run=dry_run, yes=yes)
    except Exception as exc:
        typer.echo(typer.style(str(exc), fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(out)


@app.command("validate")
def cmd_validate(
    profile: str = typer.Option("default", "--profile", "-p"),
) -> None:
    """Validate JSON files for a profile (no DB)."""
    from seed.apply import validate_profile

    errors = validate_profile(profile)
    if errors:
        for e in errors:
            typer.echo(typer.style(e, fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)
    typer.echo("OK")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
