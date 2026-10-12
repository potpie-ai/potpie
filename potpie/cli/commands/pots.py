"""Pot + source commands through Potpie-owned control-plane services."""

from __future__ import annotations

from typing import Any

import typer

from potpie.cli.commands._common import (
    EXIT_UNAVAILABLE,
    confirm_destructive_operation,
    contract,
    current_repo_identity_for_cli,
    emit,
    enrich_with_pot_guidance,
    empty_pot_warnings,
    fail,
    get_engine_client,
    current_origin,
    get_root_runtime,
    pot_directory,
    pot_graph_counts,
    pot_scope_info,
    pot_scope_resolution_human,
    repo_default_matches,
    repo_effective_pot_human,
    repo_effective_pot_info,
    repo_pot_candidates,
    resolve_pot_id,
    resolve_pot_ref,
    resolve_pot_scope,
    route_ref,
    run_engine_operation,
    use_pot_selection,
)
from potpie.cli.telemetry.onboarding_events import (
    capture_project_binding_event,
    elapsed_ms,
    now_ms,
    sanitized_failure_kind,
)
from potpie.cli.telemetry.usage_events import capture_usage_command_succeeded
from potpie.cli.repo_location import repo_identity_key, resolve_repo_location
from potpie.cli.source_kinds import (
    SourceKind,
    known_tokens,
    registrable_names,
    resolve_kind,
)
from potpie.pots.directory import is_remote_repo_key
from potpie.pots.resolution import is_archived, match_pot_ref
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.requests import ResetContextRequest

pot_app = typer.Typer(help="Pots: workspace/tenant boundaries.")
default_app = typer.Typer(help="Repo-local default pot routing.")
source_app = typer.Typer(
    help="Source registry for a pot; registration does not ingest or scan."
)
pot_app.add_typer(default_app, name="default")


def _unlistable_host_repair(origins: tuple[str, ...]) -> str:
    """Next action for a ``pot list`` in which no host answered at all.

    Names the host that is actually down: ``potpie doctor`` checks the local
    daemon, so it is a repair that cannot succeed for an unreachable managed
    service.
    """
    from potpie.config import host_registry

    local = "check backend/daemon readiness with 'potpie doctor'"
    managed = (
        "check the managed host with 'potpie host list', "
        "or re-point it with 'potpie host set <url> --token -'"
    )
    if host_registry.MANAGED not in origins:
        return local
    if host_registry.LOCAL not in origins:
        return managed
    return f"{local}; for the managed host, {managed}"


def _listed_pots(origin: str) -> tuple[list[Any], str | None]:
    """``(pots, active_pot_id)`` of one origin; raises when it cannot answer."""
    directory = pot_directory(origin)
    pots = list(directory.list_pots())
    active = directory.active_pot()
    return pots, getattr(active, "pot_id", None) if active is not None else None


@pot_app.command("list")
def pot_list(
    local: bool = typer.Option(False, "--local", help="Local-origin pots only."),
    managed: bool = typer.Option(False, "--managed", help="Managed-origin pots only."),
    all_: bool = typer.Option(
        False,
        "--all",
        help="Local + managed pots, stating a managed host that is missing or down.",
    ),
    archived: bool = typer.Option(
        False,
        "--archived",
        help="Include archived pots (hidden by default; they cannot be used).",
    ),
) -> None:
    """List pots on every configured host (narrow with --local / --managed)."""
    with contract():
        from potpie.config import host_registry

        # Default: every configured origin. The flags narrow it; naming both
        # origins (or --all) lists both, so a managed host that is missing or
        # down is stated rather than merely absent.
        wants_local = local or all_
        wants_managed = managed or all_
        if wants_local and wants_managed:
            origins: tuple[str, ...] = host_registry.ORIGINS
        elif wants_managed:
            if host_registry.managed_endpoint() is None:
                message, _ = host_registry.managed_unconfigured_refusal()
                raise CapabilityNotImplemented(
                    "host.pots.list_managed",
                    detail=message.rstrip("."),
                    recommended_next_action=(
                        "run 'potpie host set <url> --token -' (key on stdin)"
                    ),
                )
            origins = (host_registry.MANAGED,)
        elif wants_local:
            origins = (host_registry.LOCAL,)
        else:
            origins = host_registry.configured_origins()

        current = current_origin()
        rows: list[dict[str, object]] = []
        unavailable: dict[str, str] = {}
        human_lines: list[str] = []
        for origin in origins:
            if human_lines:
                human_lines.append("")
            human_lines.append(host_registry.origin_label(origin))
            try:
                pots, active_id = _listed_pots(origin)
            except Exception as exc:  # noqa: BLE001 - enumeration degrades per host
                unavailable[origin] = str(exc) or type(exc).__name__
                human_lines.append(f"  ({unavailable[origin]})")
                continue
            shown = [p for p in pots if archived or not is_archived(p)]
            hidden = len(pots) - len(shown)
            if not shown:
                human_lines.append("  (no pots)")
            for pot in shown:
                retired = is_archived(pot)
                active = bool(pot.active) or (
                    active_id is not None and pot.pot_id == active_id
                )
                # '*' is the pot commands act on; '·' is the other host's own
                # pointer, which is real but not current.
                marker = " "
                if retired:
                    marker = "~"
                elif active:
                    marker = "*" if origin == current else "·"
                rows.append(
                    {
                        "id": pot.pot_id,
                        "name": pot.name,
                        "active": active,
                        "archived": retired,
                        "current": bool(active and origin == current),
                        "origin": origin,
                    }
                )
                suffix = "  archived" if retired else ""
                human_lines.append(f"{marker} {pot.name} ({pot.pot_id}){suffix}")
            if hidden:
                human_lines.append(
                    f"  ({hidden} archived — see 'potpie pot list --archived')"
                )

        if unavailable and len(unavailable) == len(origins):
            # Degrading is only honest while some other host answered; with
            # nothing left, `pots: []` would read as "you have no pots".
            fail(
                code="unavailable",
                message=(
                    "No host answered the listing: "
                    + "; ".join(
                        f"{host_registry.origin_label(origin)}: {unavailable[origin]}"
                        for origin in origins
                    )
                ),
                detail={"unavailable": unavailable},
                next_action=_unlistable_host_repair(origins),
                exit_code=EXIT_UNAVAILABLE,
            )

        payload: dict[str, object] = {"pots": rows, "active_origin": current}
        if unavailable:
            payload["unavailable"] = unavailable
        if len(origins) > 1:
            human_lines.append("")
            human_lines.append(
                f"* active pot ({current}) · other host's active pot"
                "   — target one with '--pot <host>:<name>'"
            )
        emit(payload, human="\n".join(human_lines))


@pot_app.command("info")
def pot_info() -> None:
    with contract():
        host = get_root_runtime()
        active = pot_directory(runtime=host).active_pot()
        routing = repo_effective_pot_info(host)
        active_payload = (
            {"id": active.pot_id, "name": active.name} if active is not None else None
        )
        lines = [
            f"active: {active.name} ({active.pot_id})"
            if active is not None
            else "(no active pot)"
        ]
        routing_line = repo_effective_pot_human(routing)
        if routing_line:
            lines.append(routing_line)
        emit(
            {"active_pot": active_payload, "current_repo": routing},
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
    list_sources = getattr(pot_directory(runtime=host), "list_sources", None)
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
    if current_origin() == "managed" and not is_remote_repo_key(repo_key):
        # A local path means nothing on a managed host, and storing it there
        # would publish this machine's layout to everyone the pot is shared with.
        fail(
            code="validation_error",
            message=(
                f"'{resolved_location}' is a local path; a managed host registers "
                "repositories by their git remote."
            ),
            next_action=(
                "run it inside a clone with an 'origin' remote, or pass the "
                "remote, e.g. 'potpie source add repo github.com/<owner>/<repo>'"
            ),
        )
    repo_default_set = False
    repo_default_setter = None
    if make_default:
        pot_service = pot_directory(runtime=host)
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
    if existing is not None:
        src = existing
    else:
        src = pot_directory(runtime=host).add_source(
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
        "location": resolved_location,
        "pot_id": pot_id,
        "repo_default_set": repo_default_set,
        "repo_key": repo_key,
        "registration_only": True,
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
        pot = pot_directory(runtime=host).create_pot(name=name, use=use)
        # ``create`` reuses a live pot of the same name so ``setup`` can re-run.
        # Saying "created" for a pot that already holds a project's memory reads
        # as a fresh empty pot. A service that does not report the field keeps
        # the old wording rather than claiming a reuse it never mentioned.
        created = getattr(pot, "created", None)
        payload: dict[str, object] = {
            "id": pot.pot_id,
            "name": pot.name,
            "active": pot.active,
            "created": created,
        }
        verb = "using existing pot" if created is False else "created pot"
        human = f"{verb} '{pot.name}' ({pot.pot_id}){' [active]' if pot.active else ''}"
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
        capture_usage_command_succeeded(
            command="pot create",
            result_kind="pot_result",
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
        capture_usage_command_succeeded(
            command="pot use",
            result_kind="pot_result",
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
        pot_directory(runtime=host).set_repo_default(repo=repo_key, pot_id=pot_id)
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
        cleared = pot_directory(runtime=host).clear_repo_default(repo=repo_key)
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
        ref = route_ref(ref) or ref
        pot = pot_directory().rename_pot(ref=ref, new_name=new_name)
        emit({"id": pot.pot_id, "name": pot.name}, human=f"renamed → {pot.name}")


@pot_app.command("reset")
def pot_reset(
    ref: str = typer.Argument(None),
    confirm: bool = typer.Option(False, "--confirm"),
) -> None:
    with contract():
        host = get_root_runtime()
        pot_id = resolve_pot_id(host, ref)
        pot = next(
            item
            for item in pot_directory(runtime=host).list_pots()
            if item.pot_id == pot_id
        )
        confirmation = confirm_destructive_operation(
            confirmed_by_flag=confirm,
            prompt=(
                f"Reset all graph state and stored documents for '{pot.name}' "
                f"({pot_id})?"
            ),
            rerun_command=f"potpie pot reset {pot_id} --confirm",
        )
        result = run_engine_operation(
            get_engine_client(pot_id).reset_context(
                ResetContextRequest(),
                confirmation=confirmation,
            )
        )
        purged = _resources_purged(result)
        emit(
            {
                "id": result.context_id,
                "reset": result.reset,
                "resources_purged": purged,
            },
            human=f"reset graph state for '{pot.name}'{_teardown_suffix(purged)}",
        )


def _resources_purged(result: Any) -> bool | None:
    """The resource store's own answer from a reset, never a literal.

    ``None`` means no resource store was composed: neither a purge nor a failed
    one, so it must not print the same sentence as ``False``.
    """
    return getattr(result, "resources_purged", None)


def _teardown_suffix(purged: bool | None) -> str:
    if purged:
        return " and its stored documents"
    if purged is False:
        return " (no stored documents)"
    return ""


def _archive_target(host: Any, ref: str) -> tuple[Any, bool]:
    """``(pot, already_archived)`` for ``pot archive <ref>``.

    ``archive`` is the one command that accepts an archived pot: re-running it
    clears whatever graph state the pot still holds (a pot archived before
    archiving cleared anything keeps its data otherwise, with no command able
    to reach it). A live pot always wins the ref. An id names its pot
    exactly; a name can be shared by several archived pots, so it must then be
    narrowed to an id.
    """
    ref = route_ref(ref) or ref
    directory = pot_directory(runtime=host)
    if getattr(directory, "origin", None) == "managed":
        # Archiving clears the graph first; refusing after that would leave a
        # cleared pot that is still live, so the refusal comes before anything.
        directory.archive_pot(ref=ref)
    pots = list(directory.list_pots())
    live, archived = match_pot_ref(pots, ref)
    if live is not None:
        return live, False
    if archived is None:
        fail(
            code="pot_not_found",
            message=f"No pot matching '{ref}'.",
            next_action="run 'potpie pot list --archived'",
        )
    if archived.pot_id != ref:
        named = [p for p in pots if is_archived(p) and p.name == ref]
        if len(named) > 1:
            ids = ", ".join(p.pot_id for p in named)
            fail(
                code="ambiguous_pot",
                message=f"'{ref}' names more than one archived pot: {ids}.",
                next_action="pass the pot id: 'potpie pot archive <id> --confirm'",
            )
    return archived, True


@pot_app.command("archive")
def pot_archive(
    ref: str,
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help=(
            "Required: clears the pot's graph state and stored documents, then "
            "retires the pot."
        ),
    ),
) -> None:
    """Clear a pot's graph state and documents and retire it; it cannot be used.

    Re-running it on an archived pot clears its graph state and documents again
    and leaves it archived.
    """
    with contract():
        host = get_root_runtime()
        target, already_archived = _archive_target(host, ref)
        pot_id = target.pot_id
        label = f"'{target.name}' ({pot_id})"
        prompt = (
            f"{label} is already archived. Clear its graph state and stored "
            "documents again?"
            if already_archived
            else (
                f"Archive {label}? This clears its graph state and stored "
                "documents, and the pot cannot be used again afterwards."
            )
        )
        confirmation = confirm_destructive_operation(
            confirmed_by_flag=confirm,
            prompt=prompt,
            rerun_command=f"potpie pot archive {pot_id} --confirm",
        )
        # Clear first, flag second. The flag hides the pot from every command,
        # so flagging a pot whose graph state survived would leave data that
        # nothing can read or clear; a failed reset stops here instead.
        result = run_engine_operation(
            get_engine_client(pot_id).reset_context(
                ResetContextRequest(),
                confirmation=confirmation,
            )
        )
        purged = _resources_purged(result)
        cleared = "graph state and stored documents" if purged else "graph state"
        if already_archived:
            pot = target
            human = f"'{pot.name}' ({pot_id}) was already archived; {cleared} cleared"
        else:
            pot = pot_directory(runtime=host).archive_pot(ref=pot_id)
            human = f"archived '{pot.name}' ({cleared} cleared)"
        emit(
            {
                "id": pot.pot_id,
                "name": pot.name,
                "archived": True,
                "already_archived": already_archived,
                "graph_reset": bool(result.reset),
                "resources_purged": purged,
            },
            human=human,
        )


def _dispatch_source_kind(raw: str) -> SourceKind:
    """Resolve a kind token to its handler, or fail with the contract error.

    Both failure modes are caller mistakes (exit 1) that a retry can fix,
    and they need different retries — one points at ``resource import``,
    the other at the accepted vocabulary — so they carry distinct codes.
    """
    resolved = resolve_kind(raw)
    if resolved is None:
        fail(
            code="unknown_source_kind",
            message=f"'{raw}' is not a source kind Potpie registers.",
            detail={
                "kinds": list(registrable_names()),
                "accepted": list(known_tokens()),
            },
            next_action=("re-run with one of: " + ", ".join(registrable_names())),
        )
    if resolved.disposition == "resource":
        fail(
            code="source_kind_is_a_document",
            message=(
                f"'{raw}' names a document payload, which is stored in the "
                "resource store, not the source registry."
            ),
            detail={"kind": resolved.name},
            next_action=(
                "split the document with the matching potpie-resource-* skill, "
                "then run 'potpie resource import <dir> --doc <slug>'"
            ),
        )
    return resolved


@source_app.command("add")
def source_add(
    kind: str = typer.Argument(
        ...,
        help="repo | linear | jira | confluence | notion | url "
        "(github/gitlab/gitbucket register as repo).",
    ),
    location: str = typer.Argument(
        ...,
        help="Path, owner/repo, URL, or integration location to register. For "
        "repos, '.' or 'current' registers the current repo (resolved to its "
        "git remote or absolute path before storing).",
    ),
    name: str = typer.Option(None, "--name", help="Optional display/source name."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: resolved pot)."),
    make_default: bool | None = typer.Option(
        None,
        "--default/--no-default",
        help="Repo sources only: set this pot as the local default for this repo "
        "(on by default). Passing --default for a non-repo kind is an error.",
    ),
) -> None:
    """Register source metadata only; no ingestion or repository scan is started."""
    with contract():
        host = get_root_runtime()
        source_kind = _dispatch_source_kind(kind)
        is_repo = source_kind.disposition == "repo"
        if not is_repo and make_default is True:
            fail(
                code="repo_default_not_applicable",
                message=(
                    f"--default only applies to repo sources; '{source_kind.name}' "
                    "does not carry a repo identity."
                ),
                detail={"kind": source_kind.name},
                next_action="drop --default, or bind the pot with 'potpie pot default set'",
            )
        # Registration establishes the repo→pot mapping, so the target is the
        # explicit/active pot — never inferred from existing registrations.
        pot_id = resolve_pot_id(host, pot, infer_from_repo=False)
        started_ms = now_ms()
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_started",
            entrypoint="direct_command",
            properties={"source_kind": source_kind.name},
        )
        try:
            if is_repo:
                payload = register_repo_source(
                    host,
                    pot_id=pot_id,
                    location=location,
                    name=name,
                    make_default=make_default is not False,
                )
            else:
                src = pot_directory(runtime=host).add_source(
                    pot_id=pot_id,
                    kind=source_kind.name,
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
                    "source_kind": source_kind.name,
                    "failure_kind": sanitized_failure_kind(exc),
                    "duration_ms": elapsed_ms(started_ms),
                },
            )
            raise
        # An alias canonicalized (github → repo) is reported, never silent:
        # the stored kind is what `source list` and repo-default matching see.
        requested = kind.strip().lower()
        if requested != source_kind.name:
            payload["requested_kind"] = requested
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_completed",
            entrypoint="direct_command",
            properties={
                "source_kind": payload.get("kind", source_kind.name),
                "step_state": "done",
                "duration_ms": elapsed_ms(started_ms),
            },
        )
        capture_usage_command_succeeded(
            command="source add",
            result_kind="source_result",
        )
        resolved_location = payload.get("location", location)
        repo_default_set = bool(payload.get("repo_default_set"))
        payload, human = enrich_with_pot_guidance(
            host,
            pot_id,
            dict(payload),
            human=(
                f"registered source {payload['kind']}:{payload['name']} "
                f"({payload['source_id']}) at {resolved_location} in pot {pot_id}\n"
                + (
                    f"kind '{requested}' registered as '{source_kind.name}'\n"
                    if requested != source_kind.name
                    else ""
                )
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
        sources = pot_directory(runtime=host).list_sources(pot_id=pot_id)
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
            sources = pot_directory(runtime=host).list_sources(pot_id=pot_id)
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
            src = pot_directory(runtime=host).source_status(
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
        pot_directory(runtime=host).remove_source(pot_id=pot_id, source_id=source_id)
        emit({"removed": source_id}, human=f"removed source {source_id}")


# --- grants ----------------------------------------------------------------
#
# Sharing is a managed-host concept. The local daemon is single-user by
# construction (one implicit actor), so there is nobody to grant to, and these
# refuse there rather than pretend.


def _managed_grant_target(verb: str, ref: str) -> tuple[Any, Any]:
    """``(directory, pot)`` for a grant verb, refusing anywhere but managed."""
    host = get_root_runtime()
    resolved_ref = route_ref(ref) or ref
    if current_origin() != "managed":
        fail(
            code="unsupported",
            message=(
                f"'pot {verb}' needs a managed host: the local daemon is "
                "single-user, so a pot there has no one to share with"
            ),
            next_action="point this CLI at a service with 'potpie host set <url> --token -'",
        )
    pot = resolve_pot_ref(host, resolved_ref)
    return pot_directory(runtime=host), pot


def _grant_row(grant: Any) -> dict[str, str]:
    return {
        "pot_id": str(getattr(grant, "pot_id", "")),
        "actor_id": str(getattr(grant, "actor_id", "")),
        "role": str(getattr(grant, "role", "")),
    }


@pot_app.command("grant")
def pot_grant(
    ref: str,
    actor_id: str = typer.Argument(..., help="Actor to give access to."),
    role: str = typer.Option("writer", "--role", help="reader | writer | admin."),
) -> None:
    """Give another actor access to a pot on a managed host. Requires admin."""
    with contract():
        directory, pot = _managed_grant_target("grant", ref)
        granted = _grant_row(
            directory.grant(pot_id=pot.pot_id, actor_id=actor_id, role=role)
        )
        emit(granted, human=f"{granted['actor_id']} → {granted['role']} on '{ref}'")


@pot_app.command("revoke")
def pot_revoke(
    ref: str,
    actor_id: str = typer.Argument(..., help="Actor to remove access from."),
) -> None:
    """Take away an actor's access to a pot on a managed host. Requires admin."""
    with contract():
        directory, pot = _managed_grant_target("revoke", ref)
        revoked = bool(directory.revoke(pot_id=pot.pot_id, actor_id=actor_id))
        emit(
            {"pot": ref, "actor_id": actor_id, "revoked": revoked},
            # Not "revoked" when nothing was: that is how someone concludes an
            # actor was removed when it never had access under that name.
            human=(
                f"revoked {actor_id} from '{ref}'"
                if revoked
                else f"{actor_id} had no grant on '{ref}'"
            ),
        )


@pot_app.command("grants")
def pot_grants(ref: str) -> None:
    """List who can reach a pot on a managed host, and as what. Requires admin."""
    with contract():
        directory, pot = _managed_grant_target("grants", ref)
        grants = [_grant_row(g) for g in directory.list_grants(pot_id=pot.pot_id)]
        emit(
            {"pot": ref, "grants": grants},
            human="\n".join(f"{g['actor_id']:<20} {g['role']}" for g in grants)
            or f"no grants on '{ref}'",
        )


__all__ = ["pot_app", "source_app"]
