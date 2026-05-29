"""Per-event extraction playbooks consumed by the reconciliation agent.

Each ``EventPlaybook`` tells the agent, for a specific
``(source_system, event_type, action)`` combination, *what* this kind of event
typically carries, *what* should be extracted into the graph, and *which*
tools tend to be useful. The agent merges the playbooks for the events in a
batch into its run prompt so its behavior is shaped by the kinds of events
present, not just a static system prompt.

Lookup falls back along a ladder so callers can register narrow rules without
having to enumerate every action variant:

    (source, event_type, action) -> (source, event_type, "*") ->
    (source, "*", "*")            -> ("*", "*", "*")           -> default
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EventPlaybook:
    """Extraction guidance for one event-kind."""

    source_system: str
    event_type: str
    action: str
    summary: str
    """One-line description of what this event means."""
    available_data: str
    """What the agent can read from the event payload + via tools."""
    extract: str
    """What kinds of nodes/edges to seed (or update) in the graph."""
    skip: str = ""
    """Pitfalls — things the agent should NOT confuse for facts."""
    tool_hints: tuple[str, ...] = field(default_factory=tuple)
    """Names (or prefixes) of tools that are typically useful for this kind."""
    max_tool_calls: int = 30
    """Soft per-event tool-call budget; the per-batch budget is the max across events."""


_DEFAULT_PLAYBOOK = EventPlaybook(
    source_system="*",
    event_type="*",
    action="*",
    summary=(
        "An untyped or unfamiliar event. Read the payload and decide what, if "
        "anything, is worth recording in the graph."
    ),
    available_data=(
        "Whatever fields the producer placed under ``payload``. Use the read "
        "tools to discover what already exists before adding anything."
    ),
    extract=(
        "Be conservative. If the event clearly describes a decision, fix, "
        "incident, or owner, capture it. If unsure, add a warning instead of "
        "inventing facts."
    ),
    skip="Do not assume structure that is not present in the payload.",
    tool_hints=("context_search", "context_recent_changes"),
    max_tool_calls=15,
)


_REGISTRY: dict[tuple[str, str, str], EventPlaybook] = {}


def _register(pb: EventPlaybook) -> None:
    _REGISTRY[(pb.source_system, pb.event_type, pb.action)] = pb


# --- github / repository / added (full repo bootstrap) ---------------------
_register(
    EventPlaybook(
        source_system="github",
        event_type="repository",
        action="added",
        summary=(
            "A repository was just attached to this pot. This is the "
            "initial-bootstrap event: the graph likely contains nothing about "
            "this repo yet, so your job is to seed it with a high-level map "
            "of what the project is and what it contains."
        ),
        available_data=(
            "Payload carries owner, repo, default_branch, and maybe a "
            "remote_url. The repo is cloned into the pot's sandbox on the "
            "default branch and is reachable via sandbox_* tools. Multi-repo "
            "pots: call sandbox_list_repos first; pass repo='owner/name' on "
            "every tool. Start at the repo root and walk down."
        ),
        extract=(
            "Seed a Repository entity (entity_key ``github:repo:<owner>/<repo>``). "
            "A concrete walk that keeps you under the tool budget:\n"
            "  1. sandbox_list_repos to confirm what's attached.\n"
            "  2. sandbox_list_dir('.', repo) to see top-level layout.\n"
            "  3. sandbox_read_file('README.md', repo) — or README.rst / docs/\n"
            "     index — for purpose, audience, headline features.\n"
            "  4. Read one manifest to derive language/build/runtime:\n"
            "     package.json / pyproject.toml / Cargo.toml / go.mod / pom.xml.\n"
            "  5. sandbox_list_dir on each top-level package/module dir one\n"
            "     level deep to identify Modules and entry points.\n"
            "  6. sandbox_search('ADR', glob='*.md') and sandbox_list_dir('docs')\n"
            "     for architecture decisions / runbooks (record as Documents).\n"
            "  7. sandbox_git_log(repo, limit=20) for project age + recent\n"
            "     activity signal — seeds the first Activity entries.\n"
            "Then capture, where the walk produces evidence:\n"
            "  - the project's purpose and audience (from README / about);\n"
            "  - top-level Modules / packages / services and their roles;\n"
            "  - notable Features (canonical label ``Feature``) — user-visible "
            "    capabilities the repo exposes;\n"
            "  - entry points (CLI commands, HTTP routes, public APIs, jobs);\n"
            "  - language(s), build system, runtime, and notable dependencies;\n"
            "  - any documented architecture / ADRs / runbooks (link as Documents).\n"
            "Use stable entity_keys (e.g. ``module:<repo>:<dotted.path>``, "
            "``feature:<repo>:<slug>``) so re-runs upsert idempotently."
        ),
        skip=(
            "Do NOT enumerate every file or function — that's structural-graph "
            "territory, not the context graph. Stay at the level of features, "
            "modules, and entry points. Do NOT fabricate features that aren't "
            "evidenced by the README or code surface; add a warning instead."
        ),
        tool_hints=(
            "sandbox_list_repos",
            "sandbox_list_dir",
            "sandbox_read_file",
            "sandbox_search",
            "sandbox_git_log",
            "sandbox_git_show",
            "context_search",
            "context_graph_overview",
        ),
        max_tool_calls=120,
    )
)


# --- github / pull_request / merged ----------------------------------------
_register(
    EventPlaybook(
        source_system="github",
        event_type="pull_request",
        action="merged",
        summary=(
            "A pull request was just merged. This is a unit of completed work "
            "with a clear author, scope, and (often) a stated reason."
        ),
        available_data=(
            "PR number is in the payload; full PR metadata, commits, and "
            "review comments are reachable via the github_get_pull_request / "
            "_commits / _review_comments / _issue_comments tools. The merged "
            "diff describes the surface area changed."
        ),
        extract=(
            "Always emit one Activity (with PERFORMED + TOUCHED + IN_PERIOD). "
            "Where evidenced by the PR body, also seed: Decisions (DECIDES_FOR "
            "the affected modules / features), Fixes (RESOLVED → BugPattern or "
            "Incident), new Features (Feature → Module), and DEPENDS_ON / USES "
            "edges if dependencies were added. Link the PR as evidence."
        ),
        skip=(
            "Do not invent design decisions that the PR body does not state. "
            "Trivial PRs (typo fixes, lint) only need an Activity, no Decision."
        ),
        tool_hints=(
            "github_get_pull_request",
            "github_get_pull_request_commits",
            "github_get_pull_request_review_comments",
            "github_get_pull_request_issue_comments",
            "context_recent_changes",
        ),
        max_tool_calls=20,
    )
)


# --- github / issue / opened -----------------------------------------------
_register(
    EventPlaybook(
        source_system="github",
        event_type="issue",
        action="opened",
        summary=(
            "An issue was just filed. It may describe a bug, a feature "
            "request, or a question — read the body and labels to tell which."
        ),
        available_data=(
            "Issue number, title, body, labels, and author are in the payload "
            "(or fetchable via github_get_issue). Comments may add detail."
        ),
        extract=(
            "Emit an Activity for the filing. If the issue describes a bug, "
            "consider seeding a BugPattern or DiagnosticSignal and an Incident "
            "if user-visible. If it describes a feature request, link it to "
            "any existing Feature it touches. Always preserve the reporter as "
            "PERFORMED actor."
        ),
        skip=(
            "Don't auto-resolve issues — the open event is a signal of intent, "
            "not a confirmed bug. Use warnings for ambiguous reports."
        ),
        tool_hints=("github_get_issue", "context_search"),
        max_tool_calls=15,
    )
)


# --- manual / raw_episode / submit (UI raw ingest) -------------------------
_register(
    EventPlaybook(
        source_system="manual",
        event_type="raw_episode",
        action="submit",
        summary=(
            "A pot member submitted a free-form note or document for "
            "ingestion. The payload IS the source of truth here."
        ),
        available_data=(
            "Payload includes name, episode_body (text or url), source_description, "
            "and submitted_by_user_id. There is no external system to query — "
            "the user-supplied content is everything."
        ),
        extract=(
            "Take the user's intent at face value. Seed a Document where the "
            "note has lasting value, or a Decision / Feature / Incident if the "
            "note clearly describes one. Always emit an Activity attributing "
            "the submission to the user."
        ),
        skip=(
            "Don't try to resolve URLs or fetch external content — only the "
            "supplied text is in scope. Don't infer a code area unless the "
            "note explicitly names one."
        ),
        tool_hints=("context_search",),
        max_tool_calls=10,
    )
)


def find_playbook(
    source_system: str,
    event_type: str,
    action: str,
) -> EventPlaybook:
    """Resolve the most-specific registered playbook for an event-kind."""
    for key in (
        (source_system, event_type, action),
        (source_system, event_type, "*"),
        (source_system, "*", "*"),
        ("*", "*", "*"),
    ):
        pb = _REGISTRY.get(key)
        if pb is not None:
            return pb
    return _DEFAULT_PLAYBOOK


def all_registered_playbooks() -> list[EventPlaybook]:
    """Return every registered playbook (introspection / docs / tests)."""
    return list(_REGISTRY.values())


def render_playbooks_section(playbooks: list[EventPlaybook]) -> str:
    """Format a list of playbooks as a markdown section for the agent prompt.

    The agent receives this appended to its base instructions so it knows,
    for each event-kind in the batch, what is reachable and what to extract.
    """
    if not playbooks:
        return ""
    lines: list[str] = ["EVENT-SPECIFIC PLAYBOOKS FOR THIS BATCH:", ""]
    for pb in playbooks:
        kind_label = f"{pb.source_system} / {pb.event_type} / {pb.action}"
        lines.append(f"### Events of kind `{kind_label}`")
        lines.append(f"WHAT THIS EVENT MEANS: {pb.summary}")
        lines.append(f"WHAT YOU CAN GET: {pb.available_data}")
        lines.append(f"WHAT TO EXTRACT: {pb.extract}")
        if pb.skip:
            lines.append(f"PITFALLS: {pb.skip}")
        if pb.tool_hints:
            lines.append("USEFUL TOOLS: " + ", ".join(pb.tool_hints))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "EventPlaybook",
    "find_playbook",
    "all_registered_playbooks",
    "render_playbooks_section",
]
