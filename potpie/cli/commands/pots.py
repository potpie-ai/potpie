"""Pot + source commands through Potpie-owned control-plane services."""

from __future__ import annotations

from typing import Any

import typer

from potpie.cli.commands._common import (
    confirm_destructive_operation,
    contract,
    current_repo_identity_for_cli,
    emit,
    enrich_with_pot_guidance,
    empty_pot_warnings,
    fail,
    get_engine_client,
    get_root_runtime,
    get_pot_service,
    pot_graph_counts,
    pot_scope_info,
    pot_scope_resolution_human,
    repo_default_matches,
    repo_effective_pot_human,
    repo_effective_pot_info,
    repo_pot_candidates,
    resolve_pot_id,
    resolve_pot_scope,
    run_engine_operation,
    use_pot_selection,
)
from potpie.cli.telemetry.onboarding_events import (
    capture_project_binding_event,
    elapsed_ms,
    now_ms,
    sanitized_failure_kind,
)
from potpie.cli.repo_location import repo_identity_key, resolve_repo_location
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.requests import ResetContextRequest

pot_app = typer.Typer(help="Pots: workspace/tenant boundaries.")
default_app = typer.Typer(help="Repo-local default pot routing.")
source_app = typer.Typer(
    help="Source registry for a pot; registration does not ingest or scan."
)
pot_app.add_typer(default_app, name="default")


@pot_app.command("list")
def pot_list(
    local: bool = typer.Option(
        False, "--local", help="Local-origin pots only (default)."
    ),
    managed: bool = typer.Option(False, "--managed", help="Managed-origin pots only."),
    all_: bool = typer.Option(False, "--all", help="Local + managed pots."),
) -> None:
    with contract():
        # Managed-origin listing needs login + managed routing (HU3). '--managed'
        # alone is the structured not-implemented contract; '--all' shows locals
        # and flags managed as pending so it never crashes.
        if managed and not all_:
            raise CapabilityNotImplemented(
                "host.pots.list_managed",
                detail="managed pot listing is not implemented",
                recommended_next_action="run 'potpie login'; managed routing lands in HU3",
            )
        host = get_root_runtime()
        pots = get_pot_service(host).list_pots()
        routing = repo_effective_pot_info(host)
        effective = routing.get("effective_pot") or {}
        effective_id = effective.get("id")
        repo = routing.get("repo")
        payload: dict[str, object] = {
            "pots": [
                {
                    "id": p.pot_id,
                    "name": p.name,
                    "active": p.active,
                    "origin": "local",
                    # '*' alone hid the split the destructive ops actually use:
                    # mark the pot this working directory resolves to as well.
                    "effective_for_current_repo": bool(
                        effective_id and p.pot_id == effective_id
                    ),
                }
                for p in pots
            ],
            "current_repo": routing,
        }
        pot_rows = [
            f"{'*' if p.active else ' '}{'>' if p.pot_id == effective_id else ' '} "
            f"{p.name} ({p.pot_id})"
            for p in pots
        ]
        human_lines = ["Local", *(pot_rows or ["(no pots)"])]
        if effective_id:
            human_lines.append(
                f"  (* = active pot; > = pot commands use in {repo or 'this directory'})"
            )
        if all_:
            payload["managed_pending"] = True
            human_lines.append("  (managed pots require 'potpie login' — HU3)")
        emit(payload, human="\n".join(human_lines))


@pot_app.command("info")
def pot_info(
    pot: str = typer.Option(
        None, "--pot", help="Report this pot instead of the resolved one."
    ),
) -> None:
    with contract():
        host = get_root_runtime()
        active = get_pot_service(host).active_pot()
        routing = repo_effective_pot_info(host)
        active_payload = (
            {"id": active.pot_id, "name": active.name} if active is not None else None
        )
        pot_id, resolved_via = resolve_pot_scope(host, pot)
        selected = pot_scope_info(host, pot_id)
        repo = current_repo_identity_for_cli()
        lines = [
            f"pot: {selected['name']} ({pot_id}) "
            f"{pot_scope_resolution_human(resolved_via, repo=repo)}",
            f"  sources={selected.get('source_count', 0)} counts={selected.get('counts')}",
            f"active: {active.name} ({active.pot_id})"
            if active is not None
            else "active: (no active pot)",
        ]
        routing_line = repo_effective_pot_human(routing)
        if routing_line:
            lines.append(routing_line)
        emit(
            {
                "pot": {**selected, "resolved_via": resolved_via},
                "active_pot": active_payload,
                "current_repo": routing,
            },
            human="\n".join(lines),
        )


def _repo_key_from_option(repo: str) -> str:
    value = (repo or "").strip()
    if value in ("", ".", "current"):
        repo_key = current_repo_identity_for_cli()
    else:
        repo_key = repo_identity_key(value)
    if not repo_key:
        raise ValueError("--repo must resolve to a git remote or path")
    return repo_key


def _matching_repo_source(
    host: Any,
    *,
    pot_id: str,
    resolved_location: str,
    repo_key: str | None,
) -> Any | None:
    list_sources = getattr(get_pot_service(host), "list_sources", None)
    if not callable(list_sources):
        return None
    try:
        sources = list_sources(pot_id=pot_id)
    except Exception:  # noqa: BLE001 - duplicate detection must not block registration
        return None
    for source in sources or []:
        if getattr(source, "kind", None) != "repo":
            continue
        refs = (
            str(getattr(source, "location", "") or "").strip(),
            str(getattr(source, "name", "") or "").strip(),
        )
        if repo_key:
            if any(repo_identity_key(ref) == repo_key for ref in refs if ref):
                return source
            continue
        if resolved_location in refs:
            return source
    return None


def register_repo_source(
    host: Any,
    *,
    pot_id: str,
    location: str,
    name: str | None = None,
    make_default: bool = True,
) -> dict[str, object]:
    """Register a repo source using the same path as ``source add repo``.

    Resolves ``.`` / ``current`` to git remote or absolute path, persists
    ``location`` on the source row, and sets the repo-local default unless
    ``make_default`` is false.
    """
    resolved_location = resolve_repo_location(location)
    repo_key = repo_identity_key(resolved_location)
    repo_default_set = False
    repo_default_setter = None
    if make_default:
        pot_service = get_pot_service(host)
        if not pot_service.supports_repo_defaults:
            fail(
                code="repo_default_unavailable",
                message="This host does not support repo default bindings.",
                next_action="upgrade the local context-engine host",
            )
        repo_default_setter = pot_service.set_repo_default
    existing = _matching_repo_source(
        host,
        pot_id=pot_id,
        resolved_location=resolved_location,
        repo_key=repo_key,
    )
    reused = existing is not None
    stored_location = resolved_location
    if existing is not None:
        # Re-registration keeps the original row. Say so, and report the
        # location actually on record rather than the one just computed —
        # otherwise `source add` and `source list` disagree about where the
        # repo lives, and a passed --name looks applied when it was not.
        src = existing
        stored_location = (
            str(getattr(existing, "location", "") or "") or resolved_location
        )
    else:
        src = get_pot_service(host).add_source(
            pot_id=pot_id,
            kind="repo",
            location=resolved_location,
            name=name,
        )
    if make_default:
        if not repo_key:
            fail(
                code="repo_unresolved",
                message="Could not resolve the repository identity.",
                next_action="pass a repo location such as '<owner>/<repo>'",
            )
        repo_default_setter(repo=repo_key, pot_id=pot_id)
        repo_default_set = True
    return {
        "source_id": src.source_id,
        "kind": src.kind,
        "name": src.name,
        "location": stored_location,
        "resolved_location": resolved_location,
        "pot_id": pot_id,
        "repo_default_set": repo_default_set,
        "repo_key": repo_key,
        "registration_only": True,
        "reused_existing": reused,
        "requested_name_ignored": bool(reused and name and name != src.name),
    }


@pot_app.command("create")
def pot_create(
    name: str,
    repo: str = typer.Option(
        None,
        "--repo",
        help="Register a repo source after create (same resolution as `source add repo`).",
    ),
    use: bool = typer.Option(False, "--use"),
    no_default: bool = typer.Option(
        False,
        "--no-default",
        help="With --repo, do not set this pot as the repo-local default.",
    ),
) -> None:
    with contract():
        host = get_root_runtime()
        pot = get_pot_service(host).create_pot(name=name, use=use)
        payload: dict[str, object] = {
            "id": pot.pot_id,
            "name": pot.name,
            "active": pot.active,
        }
        human = (
            f"created pot '{pot.name}' ({pot.pot_id})"
            f"{' [active]' if pot.active else ''}"
        )
        guidance_repo: str | None = repo
        if repo is not None:
            source = register_repo_source(
                host,
                pot_id=pot.pot_id,
                location=repo,
                make_default=not no_default,
            )
            payload["source"] = source
            payload["repo_default_set"] = source["repo_default_set"]
            payload["repo_key"] = source["repo_key"]
            guidance_repo = str(source["location"])
            human = (
                f"{human}\n"
                f"registered source {source['kind']}:{source['name']} "
                f"({source['source_id']}) at {source['location']} in pot {pot.pot_id}"
            )
            if source["repo_default_set"]:
                human = f"{human}\nset repo default -> {pot.pot_id}"
            human = f"{human}\nno ingestion or scan started"
        payload, human = enrich_with_pot_guidance(
            host,
            pot.pot_id,
            payload,
            human=human,
            repo=guidance_repo,
        )
        emit(payload, human=human)


@pot_app.command("use")
def pot_use(
    ref: str,
    also_default_for_current_repo: bool = typer.Option(
        False,
        "--also-default-for-current-repo",
        help="Also set the current repo's local default pot to this pot.",
    ),
) -> None:
    with contract():
        host = get_root_runtime()
        payload, human = use_pot_selection(
            host,
            ref,
            also_default_for_current_repo=also_default_for_current_repo,
        )
        emit(payload, human=human)


@pot_app.command("linked")
def pot_linked(
    repo: str = typer.Option("current", "--repo"),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Skip per-pot graph counts for a faster repo routing summary.",
    ),
) -> None:
    """Show pots linked to a repo source and the local default, if any."""
    with contract():
        host = get_root_runtime()
        linked = repo_pot_candidates(host, repo, include_counts=not summary)
        linked["counts_included"] = not summary
        candidates = list(linked.get("candidates", ()))
        repo_key = linked.get("repo")
        lines = [f"repo {repo_key or '(unknown)'}"]
        default_id = linked.get("default_pot_id")
        if default_id:
            default = next(
                (row for row in candidates if row.get("pot_id") == default_id),
                None,
            )
            default_name = default.get("name") if default else default_id
            lines.append(f"default: {default_name} ({default_id})")
        else:
            lines.append("default: (unset)")
        if candidates:
            for row in candidates:
                counts = row.get("counts") or {}
                markers = [
                    label
                    for label, enabled in (
                        ("default", row.get("default")),
                        ("active", row.get("active")),
                    )
                    if enabled
                ]
                suffix = f"  {', '.join(markers)}" if markers else ""
                count_text = (
                    f" claims={counts.get('claims', 0)} "
                    f"entities={counts.get('entities', 0)}"
                    if not summary
                    else ""
                )
                lines.append(
                    f"  {row.get('name')} ({row.get('pot_id')}) "
                    f"sources={row.get('source_count', 0)}"
                    f"{count_text}"
                    f"{suffix}"
                )
        else:
            lines.append("  (no linked pots)")
        if summary:
            lines.append("counts omitted; rerun without --summary for graph counts")
        emit(linked, human="\n".join(lines))


@default_app.command("show")
def pot_default_show(
    repo: str = typer.Option("current", "--repo"),
    with_candidates: bool = typer.Option(
        False,
        "--with-candidates",
        help="Include the full candidates list (see `pot linked` for details).",
    ),
) -> None:
    """Show the repo-local default pot. Use --with-candidates for the full list."""
    with contract():
        host = get_root_runtime()
        linked = repo_pot_candidates(host, repo)
        default_id = linked.get("default_pot_id")
        repo_key = linked.get("repo")
        payload: dict = {
            "repo": repo_key,
            "default_pot_id": default_id,
        }
        if with_candidates:
            payload["candidates"] = linked.get("candidates", ())
        if not default_id:
            if not with_candidates:
                payload["hint"] = "run `potpie pot linked` to see all candidates"
            emit(
                payload,
                human=f"repo {repo_key or '(unknown)'} default: (unset)",
            )
            return
        info = pot_scope_info(host, default_id)
        payload["default_pot"] = info
        if not with_candidates:
            payload["hint"] = "run `potpie pot linked` to see all candidates"
        emit(
            payload,
            human=f"repo {repo_key} default: {info['name']} ({default_id})",
        )


@default_app.command("set")
def pot_default_set(ref: str, repo: str = typer.Option("current", "--repo")) -> None:
    with contract():
        host = get_root_runtime()
        pot_id = resolve_pot_id(host, ref, infer_from_repo=False)
        repo_key = _repo_key_from_option(repo)
        get_pot_service(host).set_repo_default(repo=repo_key, pot_id=pot_id)
        info = pot_scope_info(host, pot_id)
        emit(
            {"repo": repo_key, "default_pot": info},
            human=f"repo {repo_key} default → {info['name']} ({pot_id})",
        )


@default_app.command("clear")
def pot_default_clear(repo: str = typer.Option("current", "--repo")) -> None:
    with contract():
        host = get_root_runtime()
        repo_key = _repo_key_from_option(repo)
        cleared = get_pot_service(host).clear_repo_default(repo=repo_key)
        emit(
            {"repo": repo_key, "cleared": cleared},
            human=(
                f"repo {repo_key} default cleared"
                if cleared
                else f"repo {repo_key} default was not set"
            ),
        )


@pot_app.command("rename")
def pot_rename(ref: str, new_name: str) -> None:
    with contract():
        pot = get_pot_service().rename_pot(ref=ref, new_name=new_name)
        emit({"id": pot.pot_id, "name": pot.name}, human=f"renamed → {pot.name}")


@pot_app.command("reset")
def pot_reset(
    ref: str = typer.Argument(None),
    pot: str = typer.Option(
        None, "--pot", help="Pot id/name to reset (same as the positional ref)."
    ),
    confirm: bool = typer.Option(False, "--confirm"),
) -> None:
    """Clear a pot's graph state. Sources and the pot itself survive."""
    with contract():
        host = get_root_runtime()
        # An id and a name can name the *same* pot, so compare what they
        # resolve to rather than the strings the caller happened to type.
        if ref and pot and resolve_pot_id(host, ref) != resolve_pot_id(host, pot):
            fail(
                code="validation_error",
                message=(
                    f"conflicting pot selection: positional {ref!r} and "
                    f"--pot {pot!r} name different pots"
                ),
                next_action="pass the pot once, either positionally or via --pot",
            )
        selected = ref or pot
        pot_id, resolved_via = resolve_pot_scope(host, selected)
        target = pot_scope_info(host, pot_id)
        active = get_pot_service(host).active_pot()
        active_id = getattr(active, "pot_id", None) if active is not None else None
        repo = current_repo_identity_for_cli()
        # The most destructive pot operation must state exactly which pot it
        # found and how, and must not let the active-pot pointer imply a
        # different target than the one about to be cleared.
        scope_line = (
            f"'{target['name']}' ({pot_id}) "
            f"{pot_scope_resolution_human(resolved_via, repo=repo)}"
        )
        mismatch = (
            f"\nThe active pot is a different pot: {active.name} ({active_id})."
            if active_id and active_id != pot_id
            else ""
        )
        confirmation = confirm_destructive_operation(
            confirmed_by_flag=confirm,
            prompt=f"Reset all graph state for {scope_line}?{mismatch}",
            rerun_command=f"potpie pot reset {pot_id} --confirm",
            target=f"{scope_line}{mismatch}".replace("\n", " "),
        )
        result = run_engine_operation(
            get_engine_client(pot_id).reset_context(
                ResetContextRequest(),
                confirmation=confirmation,
            )
        )
        emit(
            {
                "id": result.context_id,
                "reset": result.reset,
                "pot_id": pot_id,
                "pot_name": target["name"],
                "resolved_via": resolved_via,
                "active_pot_id": active_id,
            },
            human=f"reset graph state for {scope_line}",
        )


@pot_app.command("archive")
def pot_archive(ref: str) -> None:
    with contract():
        pot = get_pot_service().archive_pot(ref=ref)
        emit({"id": pot.pot_id, "archived": True}, human=f"archived '{pot.name}'")


@source_app.command("add")
def source_add(
    kind: str = typer.Argument(..., help="repo | github | document | ..."),
    location: str = typer.Argument(
        ...,
        help="Path, owner/repo, URL, or integration location to register. For "
        "repos, '.' or 'current' registers the current repo (resolved to its "
        "git remote or absolute path before storing).",
    ),
    name: str = typer.Option(None, "--name", help="Optional display/source name."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: resolved pot)."),
    make_default: bool = typer.Option(
        True,
        "--default/--no-default",
        help="For repo sources, set this pot as the local default for this repo.",
    ),
) -> None:
    """Register source metadata only; no ingestion or repository scan is started."""
    with contract():
        host = get_root_runtime()
        source_kind = kind.strip()
        is_repo = source_kind.lower() == "repo"
        # Registration establishes the repo→pot mapping, so the target is the
        # explicit/active pot — never inferred from existing registrations.
        pot_id = resolve_pot_id(host, pot, infer_from_repo=False)
        started_ms = now_ms()
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_started",
            entrypoint="direct_command",
            properties={"source_kind": source_kind},
        )
        try:
            if is_repo:
                payload = register_repo_source(
                    host,
                    pot_id=pot_id,
                    location=location,
                    name=name,
                    make_default=make_default,
                )
            else:
                src = get_pot_service(host).add_source(
                    pot_id=pot_id,
                    kind=source_kind,
                    location=location,
                    name=name,
                )
                payload = {
                    "source_id": src.source_id,
                    "kind": src.kind,
                    "name": src.name,
                    "location": location,
                    "pot_id": pot_id,
                    "repo_default_set": False,
                    "registration_only": True,
                }
        except Exception as exc:  # noqa: BLE001
            capture_project_binding_event(
                "cli_onboarding_repo_source_add_failed",
                entrypoint="direct_command",
                properties={
                    "source_kind": kind,
                    "failure_kind": sanitized_failure_kind(exc),
                    "duration_ms": elapsed_ms(started_ms),
                },
            )
            raise
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_completed",
            entrypoint="direct_command",
            properties={
                "source_kind": payload.get("kind", source_kind),
                "step_state": "done",
                "duration_ms": elapsed_ms(started_ms),
            },
        )
        resolved_location = payload.get("location", location)
        repo_default_set = bool(payload.get("repo_default_set"))
        verb = (
            "reused existing source"
            if payload.get("reused_existing")
            else ("registered source")
        )
        notes = ""
        if payload.get("requested_name_ignored"):
            notes += (
                f"kept the existing name '{payload['name']}' "
                f"(--name is only applied at first registration)\n"
            )
        payload, human = enrich_with_pot_guidance(
            host,
            pot_id,
            dict(payload),
            human=(
                f"{verb} {payload['kind']}:{payload['name']} "
                f"({payload['source_id']}) at {resolved_location} in pot {pot_id}\n"
                + notes
                + (f"set repo default -> {pot_id}\n" if repo_default_set else "")
                + "no ingestion or scan started"
            ),
            repo=str(resolved_location) if is_repo else None,
        )
        emit(payload, human=human)


@source_app.command("list")
def source_list(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_root_runtime()
        pot_id, resolved_via = resolve_pot_scope(host, pot)
        sources = get_pot_service(host).list_sources(pot_id=pot_id)
        pot_info = pot_scope_info(host, pot_id)
        repo = (
            current_repo_identity_for_cli()
            if resolved_via in {"repo_default", "linked_repo"}
            else None
        )
        counts = pot_info.get("counts") or {}
        header = "\n".join(
            [
                (
                    f"pot={pot_info['name']} ({pot_id}) "
                    f"{pot_scope_resolution_human(resolved_via, repo=repo)}"
                ),
                (
                    f"sources={len(sources)} claims={counts.get('claims', 0)} "
                    f"entities={counts.get('entities', 0)}"
                ),
            ]
        )
        human = (
            "\n".join(
                [
                    header,
                    *(
                        f"  {s.kind}: {getattr(s, 'location', s.name)} ({s.source_id})"
                        for s in sources
                    ),
                ]
            )
            if sources
            else f"{header}\n(no sources)"
        )
        payload, human = enrich_with_pot_guidance(
            host,
            pot_id,
            {
                "pot_id": pot_id,
                "resolved_via": resolved_via,
                "repo": repo,
                "pot": pot_info,
                "source_count": len(sources),
                "sources": [
                    {
                        "id": s.source_id,
                        "kind": s.kind,
                        "name": s.name,
                        "location": getattr(s, "location", None),
                    }
                    for s in sources
                ],
            },
            human=human,
            repo=repo,
        )
        emit(payload, human=human)


def _enrich_source(host, src, pot_id: str) -> dict:
    """Build the rich source row used by both per-pot summary and single-source status."""
    location = getattr(src, "location", None)
    kind = getattr(src, "kind", "unknown")
    repo_default = False
    if kind == "repo" and location:
        repo_key = repo_identity_key(location)
        repo_default = repo_default_matches(host, repo_key, pot_id)
    return {
        "id": src.source_id,
        "kind": kind,
        "name": src.name,
        "location": location,
        "status": getattr(src, "status", "ok"),
        "repo_default": repo_default,
        "registration_only": True,
        "ingestion_status": "not_started",
    }


@source_app.command("status")
def source_status(
    source_id: str | None = typer.Argument(None),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Show source status for the pot (all sources) or a single source by ID."""
    with contract():
        host = get_root_runtime()
        pot_id = resolve_pot_id(host, pot)

        if source_id is None:
            # Per-pot summary: all sources with enriched fields
            sources = get_pot_service(host).list_sources(pot_id=pot_id)
            pot_info = pot_scope_info(host, pot_id)
            counts = pot_graph_counts(host, pot_id)
            claim_count = counts.get("claims", 0)

            source_rows = [_enrich_source(host, s, pot_id) for s in sources]

            recommended = None
            if not sources:
                recommended = (
                    "No sources registered. "
                    "Run `potpie source add repo .` to register a repository."
                )
            elif claim_count == 0:
                warnings = empty_pot_warnings(host, pot_id)
                recommended = (
                    warnings[0]
                    if warnings
                    else (
                        "Sources are registered only; no claims in graph yet. "
                        "Use ledger/agent ingestion to populate."
                    )
                )

            emit(
                {
                    "pot_id": pot_id,
                    "pot": pot_info,
                    "source_count": len(sources),
                    "claim_count": claim_count,
                    "sources": source_rows,
                    "recommended_next_action": recommended,
                },
                human=(
                    "\n".join(
                        [
                            (
                                f"pot={pot_info['name']} ({pot_id}) "
                                f"sources={len(sources)} claims={claim_count}"
                            ),
                            *(
                                (
                                    f"  {row['kind']}: "
                                    f"{row['location'] or row['name']} "
                                    f"({row['id']}) "
                                    f"status={row['status']}"
                                    + (" [repo-default]" if row["repo_default"] else "")
                                    + " [registration-only]"
                                )
                                for row in source_rows
                            ),
                        ]
                    )
                    if sources
                    else (
                        f"pot={pot_info['name']} ({pot_id}) "
                        f"sources=0 claims={claim_count}\n"
                        "(no sources)"
                    )
                )
                + (f"\nnote: {recommended}" if recommended else ""),
            )
        else:
            # Single-source mode: same enriched shape
            src = get_pot_service(host).source_status(
                pot_id=pot_id, source_id=source_id
            )
            row = _enrich_source(host, src, pot_id)
            emit(
                row,
                human=(
                    f"{src.name}: {src.status} kind={src.kind}"
                    + (" [repo-default]" if row["repo_default"] else "")
                    + " [registration-only]"
                ),
            )


@source_app.command("remove")
def source_remove(source_id: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_root_runtime()
        pot_id = resolve_pot_id(host, pot)
        get_pot_service(host).remove_source(pot_id=pot_id, source_id=source_id)
        emit({"removed": source_id}, human=f"removed source {source_id}")


__all__ = ["pot_app", "source_app"]
