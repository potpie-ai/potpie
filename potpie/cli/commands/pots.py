"""Pot + source commands → ``HostShell.pots`` (PotManagementService)."""

from __future__ import annotations

from typing import Any, NoReturn

import typer

from potpie.cli.commands._common import (
    contract,
    current_repo_identity_for_cli,
    emit,
    enrich_with_pot_guidance,
    empty_pot_warnings,
    fail,
    get_host,
    get_host_for,
    invalidate_host_snapshot,
    pot_graph_counts,
    pot_scope_info,
    pot_scope_resolution_human,
    repo_default_matches,
    repo_effective_pot_human,
    repo_effective_pot_info,
    repo_pot_candidates,
    resolve_pot_id,
    resolve_pot_scope,
    use_pot_selection,
)
from potpie.cli.telemetry.onboarding_events import (
    capture_project_binding_event,
    elapsed_ms,
    now_ms,
    sanitized_failure_kind,
)
from potpie.cli.repo_location import (
    repo_identity_key_for_location,
    resolve_repo_location,
)
from potpie.cli.source_kinds import (
    SourceKind,
    known_tokens,
    registrable_names,
    resolve_kind,
)
from potpie_context_core.errors import CapabilityNotImplemented, SourceNotFound
from potpie_context_engine.domain.ports.services.pot_management import (
    INGESTION_NOT_STARTED,
    SOURCE_REGISTERED,
)

pot_app = typer.Typer(help="Pots: workspace/tenant boundaries.")
default_app = typer.Typer(help="Repo-local default pot routing.")
source_app = typer.Typer(
    help="Source registry for a pot; registration does not ingest or scan."
)
pot_app.add_typer(default_app, name="default")


def _unlistable_host_repair(origins: tuple[str, ...]) -> str:
    """Next action for a ``pot list`` in which no host answered at all.

    Names the host that is actually down instead of always pointing at
    ``potpie doctor``: doctor only knows about the local daemon, so handing it
    to someone whose *managed* service is unreachable is a repair that cannot
    succeed — the same misdirection as sending an operator to ``pot list`` to
    hunt for a pot that was never missing.
    """
    from potpie.cli import hosts

    local = "check backend/daemon readiness with 'potpie doctor'"
    managed = (
        "check the managed host with 'potpie host list', "
        "or re-point it with 'potpie host set <base-url>'"
    )
    if hosts.MANAGED not in origins:
        return local
    if hosts.LOCAL not in origins:
        return managed
    return f"{local}; for the managed host, {managed}"


@pot_app.command("list")
def pot_list(
    local: bool = typer.Option(
        False, "--local", help="Local-origin pots only (default)."
    ),
    managed: bool = typer.Option(False, "--managed", help="Managed-origin pots only."),
    all_: bool = typer.Option(False, "--all", help="Local + managed pots."),
    archived: bool = typer.Option(
        False,
        "--archived",
        help="Include archived pots (excluded by default; they cannot be used).",
    ),
) -> None:
    with contract():
        from potpie.cli import hosts

        # Default shows every configured origin — the question "which pots do I
        # have" is not usefully answered per host, and a bare `pot list` that
        # hid the managed ones is what sent people looking for a broken server.
        # The flags narrow it; --all forces both sections so an unconfigured or
        # unreachable managed host is stated rather than merely absent.
        #
        # A flag combination that names *both* origins is honoured as both,
        # never as whichever branch happened to be tested first: `--local
        # --managed` used to fall through to the configured origins, so on the
        # default install it printed a local-only listing at exit 0 while
        # `--managed` on its own refused — half the request dropped without a
        # word. Same for `--all --local`, which is `--all` with one origin
        # named twice, not a narrowing of it.
        wants_local = local or all_
        wants_managed = managed or all_
        if wants_local and wants_managed:
            origins: tuple[str, ...] = hosts.ORIGINS
        elif wants_managed:
            if hosts.managed_endpoint() is None:
                raise CapabilityNotImplemented(
                    "host.pots.list_managed",
                    detail="no managed host is configured",
                    recommended_next_action="run 'potpie host set <base-url> [--token <key>]'",
                )
            origins = (hosts.MANAGED,)
        elif wants_local:
            origins = (hosts.LOCAL,)
        else:
            origins = hosts.configured_origins()

        current = hosts.current_origin()
        rows: list[dict[str, object]] = []
        unavailable: dict[str, str] = {}
        human_lines: list[str] = []

        for origin in origins:
            if human_lines:
                human_lines.append("")
            human_lines.append(hosts.origin_label(origin))
            try:
                pots = get_host_for(origin).pots.list_pots()
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                # Enumeration degrades: a listing that dies because a remote is
                # down is useless exactly when you need to see what is local.
                unavailable[origin] = str(exc)
                # The message already names the host it came from, and the
                # section header names it too; a third "unavailable:" prefix
                # would say it three times in one line.
                human_lines.append(f"  ({exc})")
                continue
            # Archived pots are hidden rather than listed unmarked. Every pot
            # command refuses them, so leaving them in the default listing
            # offered targets that nothing accepts — and `archived` was not even
            # in the JSON, so no consumer could tell them apart.
            shown = [p for p in pots if archived or not getattr(p, "archived", False)]
            hidden = len(pots) - len(shown)
            if not shown:
                human_lines.append("  (no pots)")
            for pot in shown:
                is_archived = bool(getattr(pot, "archived", False))
                # '*' is the pot commands actually act on; '·' is the other
                # host's own pointer, which is real but not current.
                marker = " "
                if is_archived:
                    marker = "~"
                elif pot.active:
                    marker = "*" if origin == current else "·"
                rows.append(
                    {
                        "id": pot.pot_id,
                        "name": pot.name,
                        "active": bool(pot.active),
                        "archived": is_archived,
                        "current": bool(pot.active and origin == current),
                        "origin": origin,
                    }
                )
                suffix = "  archived" if is_archived else ""
                human_lines.append(f"{marker} {pot.name} ({pot.pot_id}){suffix}")
            if hidden:
                human_lines.append(
                    f"  ({hidden} archived — see 'potpie pot list --archived')"
                )

        if unavailable and len(unavailable) == len(origins):
            # Degrading is only honest while some *other* host answered. With
            # nothing left to degrade to — the sole configured host is down, or
            # `--local`/`--managed` targeted exactly the host that is down — the
            # envelope below is `pots: []`, and no `--json` consumer can tell
            # that apart from "you have no pots": `pot list | jq '.pots|length'`
            # answers 0 either way, and `unavailable` is a key nobody reads
            # because exit 0 said there was nothing to read. Before the per-origin
            # catch existed this propagated and exited 2, which is the answer.
            fail(
                code="unavailable",
                message=(
                    "No host answered the listing: "
                    + "; ".join(
                        f"{hosts.origin_label(origin)}: {unavailable[origin]}"
                        for origin in origins
                    )
                ),
                detail={"unavailable": unavailable},
                next_action=_unlistable_host_repair(origins),
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
        host = get_host()
        active = host.pots.active_pot()
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
        # The same key `source add repo --default` writes, so binding a repo by
        # path and reading it back from inside that path cannot disagree.
        repo_key = repo_identity_key_for_location(value)
    if not repo_key:
        raise ValueError("--repo must resolve to a git remote or path")
    return repo_key


def _existing_source(
    host: Any,
    *,
    pot_id: str,
    kind: str,
    resolved_location: str,
    repo_key: str | None,
) -> Any | None:
    """The row this registration would duplicate, or ``None``.

    Repo sources are matched by repo identity — the point of the key is that
    ``.``, an absolute path and ``git@github.com:acme/shop.git`` are one
    repository. Every other kind is matched on its stored location, which is
    all the identity a Linear team or a URL has: ``source add linear team/PLAT``
    run twice used to leave two rows, and a registry that answers "which
    sources does this pot have" with the same source twice is wrong in the
    listing, in ``source_count``, and in whatever a harness does per source.
    """
    list_sources = getattr(host.pots, "list_sources", None)
    if not callable(list_sources):
        return None
    try:
        sources = list_sources(pot_id=pot_id)
    except Exception:  # noqa: BLE001 - duplicate detection must not block registration
        return None
    for source in sources or []:
        if getattr(source, "kind", None) != kind:
            continue
        refs = (
            str(getattr(source, "location", "") or "").strip(),
            str(getattr(source, "name", "") or "").strip(),
        )
        if kind == "repo" and repo_key:
            if any(
                repo_identity_key_for_location(ref) == repo_key for ref in refs if ref
            ):
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

    Everything that can refuse this registration refuses it *before* the row is
    written. The repo-identity check used to run the other way round: an
    unresolvable location was persisted as a source row and only then failed,
    so exit 1 left a junk registration behind — and because dedup then matched
    that row, every retry "found" it and never wrote a real one.
    """
    resolved_location = resolve_repo_location(location)
    repo_key = repo_identity_key_for_location(resolved_location)
    repo_default_set = False
    repo_default_setter = None
    if make_default:
        repo_default_setter = getattr(host.pots, "set_repo_default", None)
        if not callable(repo_default_setter):
            fail(
                code="repo_default_unavailable",
                message="This host does not support repo default bindings.",
                next_action="upgrade the local context-engine host",
            )
        if not repo_key:
            fail(
                code="repo_unresolved",
                message="Could not resolve the repository identity.",
                next_action="pass a repo location such as '<owner>/<repo>'",
            )
    existing = _existing_source(
        host,
        pot_id=pot_id,
        kind="repo",
        resolved_location=resolved_location,
        repo_key=repo_key,
    )
    if existing is not None:
        src = existing
    else:
        src = host.pots.add_source(
            pot_id=pot_id,
            kind="repo",
            location=resolved_location,
            name=name,
        )
        invalidate_host_snapshot()
    if make_default:
        try:
            repo_default_setter(repo=repo_key, pot_id=pot_id)
        except Exception:
            # The binding is the reason `source add repo` is more than a row, so
            # a row that outlives the failure of the thing it was written for is
            # the same junk registration — and the one every later retry would
            # then dedup against.
            _discard_registration(
                host, pot_id=pot_id, source=src, created=existing is None
            )
            raise
        invalidate_host_snapshot()
        repo_default_set = True
    payload: dict[str, object] = {
        "source_id": src.source_id,
        "kind": src.kind,
        "name": src.name,
        "location": resolved_location,
        "pot_id": pot_id,
        "repo_default_set": repo_default_set,
        "repo_key": repo_key,
        "registration_only": True,
    }
    # Reported only when it happened, like ``requested_kind``: "we reused the
    # row you already had" is the one thing this answer would otherwise hide
    # behind a fresh-looking success.
    if existing is not None:
        payload["already_registered"] = True
    return payload


def _discard_registration(
    host: Any, *, pot_id: str, source: Any, created: bool
) -> None:
    """Undo a row this invocation wrote, when the rest of the command failed.

    Only a row *this* invocation created is dropped: reusing an existing
    registration and then failing must not delete the registration that was
    already there.
    """
    if not created:
        return
    remover = getattr(host.pots, "remove_source", None)
    if not callable(remover):
        return
    try:
        remover(pot_id=pot_id, source_id=source.source_id)
    except Exception:  # noqa: BLE001, S110 - the original failure is the one to report
        pass


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
        host = get_host()
        pot = host.pots.create_pot(name=name, use=use)
        invalidate_host_snapshot()
        # ``create`` is idempotent so ``setup`` can re-run, but saying "created"
        # for a pot that already held a project's memory reads as a fresh empty
        # pot, which is the opposite of what was returned. A host that does not
        # report the field keeps the old wording rather than being made to claim
        # a reuse it never mentioned — see ``PotInfo.created``.
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
        host = get_host()
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
        host = get_host()
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
        host = get_host()
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
        host = get_host()
        pot_id = resolve_pot_id(host, ref, infer_from_repo=False)
        repo_key = _repo_key_from_option(repo)
        host.pots.set_repo_default(repo=repo_key, pot_id=pot_id)
        invalidate_host_snapshot()
        info = pot_scope_info(host, pot_id)
        emit(
            {"repo": repo_key, "default_pot": info},
            human=f"repo {repo_key} default → {info['name']} ({pot_id})",
        )


@default_app.command("clear")
def pot_default_clear(repo: str = typer.Option("current", "--repo")) -> None:
    with contract():
        host = get_host()
        repo_key = _repo_key_from_option(repo)
        cleared = host.pots.clear_repo_default(repo=repo_key)
        invalidate_host_snapshot()
        emit(
            {"repo": repo_key, "cleared": cleared},
            human=(
                f"repo {repo_key} default cleared"
                if cleared
                else f"repo {repo_key} default was not set"
            ),
        )


#: "the result carried no teardown answer at all", which is not the same value
#: as ``resources_purged=None`` and must not collapse into it — see
#: :func:`_teardown`.
_TEARDOWN_UNSAID: Any = object()


def _teardown(result: Any) -> tuple[Any, bool | None, bool]:
    """``(pot, resources_purged, reported)`` from a reset/archive result.

    A current host returns ``PotTeardownResult`` — the pot plus what tearing
    its data down actually did. An older one returns the bare ``PotInfo``,
    which says the call succeeded and nothing at all about teardown. Reading
    ``.pot`` off that raised ``AttributeError``, and the error boundary
    flattened it into "Unexpected internal error" *after* the host had already
    archived the pot: a destructive command reporting failure on success is the
    worst shape this could take, because the obvious response is to run it
    again.

    ``reported`` is kept separate from ``resources_purged`` rather than folded
    into its ``None``: that ``None`` already means "no resource store was
    wired", which is a fact about the pot. Not knowing is a fact about the
    host, and the two must not print the same sentence — one of them is
    entitled to claim a teardown happened and the other is not.

    Which is why the second read defaults to :data:`_TEARDOWN_UNSAID` and not to
    ``None``: defaulting to ``None`` re-merged, one line later, the two answers
    the split exists to keep apart. A shape carrying ``.pot`` and no
    ``resources_purged`` — neither of the two shipped ones, but every future
    one is a candidate — came back as ``reported=True`` and printed "(no stored
    resources)", a claim about the pot that the host never made.
    """
    pot = getattr(result, "pot", None)
    if pot is None:
        return result, None, False
    purged = getattr(result, "resources_purged", _TEARDOWN_UNSAID)
    if purged is _TEARDOWN_UNSAID:
        return pot, None, False
    return pot, purged, True


_TEARDOWN_UNREPORTED = (
    "this host runs an older contract and reported no teardown detail; "
    "verify with 'potpie graph status'"
)


#: ``pot reset``/``archive``/``rename`` were the only pot commands that handed a
#: raw ref straight to the host. Two consequences, both bad on the commands whose
#: job is destroying data: ``pot reset managed:x`` failed as ``pot_not_found``
#: even though ``pot list`` prints that exact syntax as the way to target a host,
#: and a bare name matching a pot on both hosts picked one silently. Routing them
#: through the same resolver as ``--pot`` gives them the qualifier, the
#: cross-host ambiguity refusal, and the origin move — see
#: :func:`_common._resolve_explicit_pot`.
def _destructive_target(host: Any, ref: str) -> tuple[str, str]:
    """``(pot_id, label)`` for a ref a destructive command is about to act on."""
    pot_id = resolve_pot_id(host, ref, infer_from_repo=False)
    return pot_id, _target_label(host, pot_id)


def _target_label(host: Any, pot_id: str) -> str:
    """How to name the target in a confirmation prompt: name, id *and* host.

    A prompt that says only ``resetting 'default'`` is not enough to consent to
    when ``default`` exists on two hosts and the CLI has been moved to one of
    them by the ref itself. The *name* is here for the mirror-image case: a
    target the caller did not type — a bare ``pot reset`` resolved off the repo
    binding — arrives as an opaque ``pot_37ab7f7fe4ef``, which nobody can match
    against the pot they believe they are working in.
    """
    from potpie.cli import hosts

    name = _pot_name(host, pot_id)
    named = f"'{name}' ({pot_id})" if name else pot_id
    return f"{named} on the {hosts.current_origin()} host"


def _pot_name(host: Any, pot_id: str) -> str | None:
    """The pot's display name, or ``None`` if the host will not say.

    Naming the target is an improvement to a refusal or a prompt, never a
    precondition for it: a host that cannot enumerate must still be able to
    print "resetting pot_x" rather than dying inside the message it was
    building.
    """
    try:
        pots = host.pots.list_pots()
    except Exception:  # noqa: BLE001 - see docstring
        return None
    for pot in pots or ():
        if getattr(pot, "pot_id", None) == pot_id:
            return getattr(pot, "name", None)
    return None


@pot_app.command("rename")
def pot_rename(ref: str, new_name: str) -> None:
    with contract():
        host = get_host()
        pot_id, _ = _destructive_target(host, ref)
        pot = host.pots.rename_pot(ref=pot_id, new_name=new_name)
        invalidate_host_snapshot()
        emit({"id": pot.pot_id, "name": pot.name}, human=f"renamed → {pot.name}")


def _inferred_reset_target(host: Any) -> tuple[str, str]:
    """``(pot_id, label)`` for a ``pot reset`` that named no pot at all.

    The pot still comes out of :func:`resolve_pot_scope` — the same resolution
    every other command uses, repo default then the repo's single linked pot
    then the active pot — so a bare ``reset`` can never wipe a pot that is not
    the one the rest of the CLI is reading and writing. That order is right for
    a read and not sufficient for a wipe: it *finds* a pot rather than being
    told one. With a repo default pointing at one pot and a different pot
    active, ``pot reset --confirm`` destroyed the repo default while ``pot
    info`` named the other as active; with no active pot at all it still found
    the repo's pot and destroyed that.

    So inference may find the target and may not choose it: unless the resolved
    pot is also the active one, the caller has to name it. Typing the ref costs
    a word; re-creating a wiped pot's memory is not possible at all.
    """
    target, resolved_via = resolve_pot_scope(host)
    label = _target_label(host, target)
    active = host.pots.active_pot()
    active_id = getattr(active, "pot_id", None)
    if active_id == target:
        return target, label
    repo = (
        current_repo_identity_for_cli()
        if resolved_via in {"repo_default", "linked_repo"}
        else None
    )
    via = pot_scope_resolution_human(resolved_via, repo=repo)
    detail = {
        "target_pot_id": target,
        "resolved_via": resolved_via,
        "active_pot_id": active_id,
    }
    next_action = (
        f"name the pot you mean: 'potpie pot reset {target} --confirm', "
        f"or select it first with 'potpie pot use {target}'"
    )
    if active_id is None:
        fail(
            code="no_active_pot",
            message=(
                f"No pot is active, so 'potpie pot reset' will not pick one: it "
                f"would have wiped {label}, {via}."
            ),
            detail=detail,
            next_action=next_action,
        )
    fail(
        code="ambiguous_pot",
        message=(
            f"'potpie pot reset' with no ref would wipe {label}, {via} — but the "
            f"active pot is '{getattr(active, 'name', active_id)}' ({active_id})."
        ),
        detail=detail,
        next_action=next_action,
    )


@pot_app.command("reset")
def pot_reset(
    ref: str = typer.Argument(None),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Required: clears the pot's graph partition and resource store.",
    ),
) -> None:
    with contract():
        host = get_host()
        if ref:
            target, label = _destructive_target(host, ref)
        else:
            target, label = _inferred_reset_target(host)
        if not confirm:
            fail(
                code="confirmation_required",
                message=(
                    f"resetting {label} clears its graph and stored document chunks"
                ),
                next_action=f"re-run with 'potpie pot reset {target} --confirm'",
            )
        # ``resources_purged`` is the store's own answer, not a literal: it
        # used to report a cleared resource tree on pots that never held one.
        pot, purged, reported = _teardown(
            host.pots.reset_pot(ref=target, confirm=confirm)
        )
        invalidate_host_snapshot()
        if not reported:
            human = f"reset '{pot.name}' — {_TEARDOWN_UNREPORTED}"
        elif purged:
            human = f"reset graph and resources for '{pot.name}'"
        else:
            human = f"reset graph for '{pot.name}' (no stored resources)"
        emit(
            {
                "id": pot.pot_id,
                "reset": True,
                "resources_purged": purged,
                "teardown_reported": reported,
            },
            human=human,
        )


@pot_app.command("archive")
def pot_archive(
    ref: str,
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Required: archives the pot and tears down its graph and resources.",
    ),
) -> None:
    with contract():
        host = get_host()
        target, label = _destructive_target(host, ref)
        if not confirm:
            fail(
                code="confirmation_required",
                message=(
                    f"archiving {label} also clears its graph and stored document "
                    "chunks, and the pot cannot be used again afterwards"
                ),
                next_action=f"re-run with 'potpie pot archive {target} --confirm'",
            )
        pot, purged, reported = _teardown(host.pots.archive_pot(ref=target))
        invalidate_host_snapshot()
        if not reported:
            # Deliberately does not say "graph cleared". On a host that predates
            # the teardown contract, archive can be a flag and nothing more —
            # the managed service today leaves every claim in place — so
            # claiming the graph was cleared would be a lie told by the one
            # command whose whole job is destroying data.
            human = f"archived '{pot.name}' — {_TEARDOWN_UNREPORTED}"
        elif purged:
            human = f"archived '{pot.name}' (graph and resources cleared)"
        else:
            human = f"archived '{pot.name}' (graph cleared; no stored resources)"
        emit(
            {
                "id": pot.pot_id,
                "archived": True,
                "resources_purged": purged,
                "teardown_reported": reported,
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


def _require_location(location: str, *, kind: SourceKind) -> str:
    """The location to register, refusing one that identifies nothing.

    ``source add linear '   '`` exited 0 and left a row whose location, name and
    every later match against it were blank: un-routable, un-ingestable, and
    visible only as an extra number in ``source_count``. Refused at the boundary
    the user typed at, so the message can name the argument; the control plane
    refuses it again, because a registry that accepts a nameless row is wrong
    whoever asked.
    """
    cleaned = (location or "").strip()
    if cleaned:
        return cleaned
    fail(
        code="missing_source_location",
        message=(
            f"A '{kind.name}' source needs a location; an empty one registers nothing."
        ),
        detail={"kind": kind.name},
        next_action=(
            "pass what to register, e.g. 'potpie source add repo .'"
            if kind.disposition == "repo"
            else f"pass what to register, e.g. 'potpie source add {kind.name} <ref>'"
        ),
    )


def _fail_source_not_found(exc: SourceNotFound, *, pot_id: str) -> NoReturn:
    """Report a missing *source* as one, with the listing that would find it.

    ``source_not_found``, never ``pot_not_found``: the pot resolved fine, and
    sending the operator to ``pot list`` to hunt for a pot that is not missing
    is a repair that cannot succeed. The registration is usually alive in the
    pot they did not pass.
    """
    fail(
        code="source_not_found",
        message=str(exc),
        detail={"pot_id": pot_id},
        next_action=(
            getattr(exc, "recommended_next_action", None)
            or f"list this pot's sources with 'potpie source list --pot {pot_id}'"
        ),
    )


def _ingestion_marker(row: dict) -> str:
    """The human suffix for how far ingestion has got on a source row."""
    if row.get("registration_only"):
        return " [registration-only]"
    return f" [ingestion: {row.get('ingestion_status')}]"


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
        host = get_host()
        source_kind = _dispatch_source_kind(kind)
        location = _require_location(location, kind=source_kind)
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
                # Idempotent for every kind, not just repo: re-running
                # `source add linear team/PLAT` doubled the registry, and each
                # copy then counted as a source of its own everywhere sources
                # are counted or walked.
                existing = _existing_source(
                    host,
                    pot_id=pot_id,
                    kind=source_kind.name,
                    resolved_location=location,
                    repo_key=None,
                )
                src = existing or host.pots.add_source(
                    pot_id=pot_id,
                    kind=source_kind.name,
                    location=location,
                    name=name,
                )
                invalidate_host_snapshot()
                payload = {
                    "source_id": src.source_id,
                    "kind": src.kind,
                    "name": src.name,
                    "location": location,
                    "pot_id": pot_id,
                    "repo_default_set": False,
                    "registration_only": True,
                }
                if existing is not None:
                    payload["already_registered"] = True
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
        host = get_host()
        pot_id, resolved_via = resolve_pot_scope(host, pot)
        sources = host.pots.list_sources(pot_id=pot_id)
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
    """Build the rich source row used by both per-pot summary and single-source status.

    Every field here is the row's own answer. Three of them were literals —
    ``status`` fell back to ``"ok"``, ``ingestion_status`` was always
    ``"not_started"`` and ``registration_only`` was always ``True`` — so
    ``source status`` reported a healthy, never-ingested registration for a
    source stored as ``error`` and ingested last week. A status command that
    cannot report a bad status is a status command with nothing to say.
    """
    location = getattr(src, "location", None)
    kind = getattr(src, "kind", "unknown")
    repo_default = False
    if kind == "repo" and location:
        repo_key = repo_identity_key_for_location(location)
        repo_default = repo_default_matches(host, repo_key, pot_id)
    ingestion_status = str(
        getattr(src, "ingestion_status", None) or INGESTION_NOT_STARTED
    )
    return {
        "id": src.source_id,
        "kind": kind,
        "name": src.name,
        "location": location,
        "status": str(getattr(src, "status", None) or SOURCE_REGISTERED),
        "repo_default": repo_default,
        # Derived on the row when the host reports one (``SourceInfo`` computes
        # it from the ingestion status); otherwise from the ingestion status
        # this row carries, so an ingested source is never labelled
        # registration-only.
        "registration_only": bool(
            getattr(src, "registration_only", ingestion_status == INGESTION_NOT_STARTED)
        ),
        "ingestion_status": ingestion_status,
    }


@source_app.command("status")
def source_status(
    source_id: str | None = typer.Argument(None),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Show source status for the pot (all sources) or a single source by ID."""
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)

        if source_id is None:
            # Per-pot summary: all sources with enriched fields
            sources = host.pots.list_sources(pot_id=pot_id)
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
                                    + _ingestion_marker(row)
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
            try:
                src = host.pots.source_status(pot_id=pot_id, source_id=source_id)
            except SourceNotFound as exc:
                _fail_source_not_found(exc, pot_id=pot_id)
            row = _enrich_source(host, src, pot_id)
            emit(
                row,
                human=(
                    f"{src.name}: {row['status']} kind={row['kind']}"
                    + (" [repo-default]" if row["repo_default"] else "")
                    + _ingestion_marker(row)
                ),
            )


@source_app.command("remove")
def source_remove(source_id: str, pot: str = typer.Option(None, "--pot")) -> None:
    """Drop a source registration. Does not purge documents or graph claims."""
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        removed = host.pots.remove_source(pot_id=pot_id, source_id=source_id)
        invalidate_host_snapshot()
        # Only an explicit ``False`` is a miss — a host that answers nothing is
        # on an older contract and cannot be accused of having removed nothing.
        # Reporting "removed source <id>" for an id this pot never held is the
        # worst possible answer to the likeliest mistake: the registration is
        # usually alive in the pot the caller did not pass, and the message
        # sends them away believing it is gone. Named like the same miss in
        # ``source status`` so both commands answer it identically — same error
        # type, same code, same repair.
        if removed is False:
            _fail_source_not_found(
                SourceNotFound(f"No source '{source_id}' in pot '{pot_id}'."),
                pot_id=pot_id,
            )
        emit(
            {"removed": source_id, "resources_touched": False},
            human=(
                f"removed source {source_id} (registration only; documents unchanged)"
            ),
        )


__all__ = ["pot_app", "source_app"]


# --- grants ----------------------------------------------------------------
#
# Sharing is a *managed host* concept. The local daemon is single-user by
# construction — no identity on the wire, one implicit actor — so there is
# nobody to grant to, and these three refuse there rather than dispatching a
# member the daemon has never heard of and reporting whatever it says.


def _managed_only(verb: str) -> None:
    """Refuse a grant verb anywhere but a managed host, and say why."""
    from potpie.cli import hosts

    origin = hosts.current_origin()
    if origin != "managed":
        fail(
            code="unsupported",
            message=(
                f"'pot {verb}' needs a managed host: the local daemon is "
                "single-user, so a pot there has no one to share with"
            ),
            next_action="point this CLI at a service with 'potpie host set <url>'",
        )


@pot_app.command("grant")
def pot_grant(
    ref: str,
    actor_id: str = typer.Argument(..., help="Actor to give access to."),
    role: str = typer.Option("writer", "--role", help="reader | writer | admin."),
) -> None:
    """Give another actor access to a pot. Requires admin on it."""
    with contract():
        _managed_only("grant")
        host = get_host()
        granted = host.pots.grant_pot(ref=ref, actor_id=actor_id, role=role)
        invalidate_host_snapshot()
        emit(
            dict(granted),
            human=f"{granted['actor_id']} → {granted['role']} on '{ref}'",
        )


@pot_app.command("revoke")
def pot_revoke(
    ref: str,
    actor_id: str = typer.Argument(..., help="Actor to remove access from."),
) -> None:
    """Take away an actor's access to a pot. Requires admin on it."""
    with contract():
        _managed_only("revoke")
        host = get_host()
        revoked = bool(host.pots.revoke_pot(ref=ref, actor_id=actor_id))
        invalidate_host_snapshot()
        emit(
            {"pot": ref, "actor_id": actor_id, "revoked": revoked},
            # "no grant to remove" rather than "revoked", because reporting a
            # revocation that did not happen is how someone concludes an actor
            # has been removed when they never had access under that name.
            human=(
                f"revoked {actor_id} from '{ref}'"
                if revoked
                else f"{actor_id} had no grant on '{ref}'"
            ),
        )


@pot_app.command("grants")
def pot_grants(ref: str) -> None:
    """List who can reach a pot, and as what. Requires admin on it."""
    with contract():
        _managed_only("grants")
        host = get_host()
        grants = [dict(g) for g in host.pots.list_grants(ref=ref)]
        emit(
            {"pot": ref, "grants": grants},
            human="\n".join(f"{g['actor_id']:<20} {g['role']}" for g in grants)
            or f"no grants on '{ref}'",
        )
