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
from pathlib import Path

_PLAYBOOKS_DIR = Path(__file__).resolve().parent / "playbooks"


def _load_skill_body(filename: str) -> str:
    """Return the markdown body of a skill file under ``domain/playbooks/``.

    Strips the YAML frontmatter (``---`` … ``---``) so the body is ready to
    drop into a playbook's ``extract`` field, which ``render_playbooks_section``
    embeds directly into the agent prompt.
    """
    path = _PLAYBOOKS_DIR / filename
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            f"Skill body {filename!r} is unavailable. Stop and report this "
            "configuration error instead of attempting this playbook."
        )
    if not raw.startswith("---\n"):
        return raw
    end = raw.find("\n---\n", 4)
    if end < 0:
        return raw
    return raw[end + 5 :].lstrip()


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
    enables_planner: bool = False
    """Turn on the agent's built-in todo/plan tools for batches containing this
    event-kind. Backfill-style seeds (a single ``*.added`` event whose handling
    fans out into many artifacts) need the planner so the agent can durably
    track an enumerate-then-drain todo list across checkpoint resumes. Normal
    live event-kinds leave this ``False`` — they're small and the planning
    overhead isn't worth the tokens."""


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
    tool_hints=("context_search", "context_timeline"),
    max_tool_calls=15,
)


_REGISTRY: dict[tuple[str, str, str], EventPlaybook] = {}


def _register(pb: EventPlaybook) -> None:
    _REGISTRY[(pb.source_system, pb.event_type, pb.action)] = pb


# --- github / repository / added (source-history seed) ---------------------
_register(
    EventPlaybook(
        source_system="github",
        event_type="repository",
        action="added",
        summary=(
            "A repository was just attached to this pot. This event registers "
            "the source and may seed authored source history, but it is not a "
            "working-tree scan or module-map bootstrap."
        ),
        available_data=(
            "Payload carries owner, repo, default_branch, and maybe a "
            "remote_url. GitHub list/get tools can enumerate recent merged "
            "PRs and standalone issues. If the payload includes explicit "
            "documents or links, read those specific sources; do not walk the "
            "repository file tree."
        ),
        extract=(
            "Your todo/plan tools are ON for this batch. Use them for a "
            "bounded source-history seed: enumerate, write one todo per "
            "artifact, then drain the list. A resumed run CONTINUES the todo "
            "list instead of re-enumerating.\n\n"
            "PHASE 1 — source registration: ensure the Repository entity is "
            "represented with the attached owner/repo metadata. Do not infer "
            "services, modules, dependencies, features, or architecture from "
            "the file tree.\n\n"
            "PHASE 2 — historical backfill: use github_list_pull_requests(repo) "
            "and github_list_issues(repo). These are bounded server-side to a "
            "trailing window and hard item cap and come back newest-first. "
            "ONE call each returns compact refs; do NOT page beyond what they "
            "return. Drain PRs first, then issues. For each ref, hydrate with "
            "github_get_pull_request / github_get_issue and supporting PR "
            "metadata tools only where the per-kind playbooks below say it is "
            "worth it. Then apply_graph_mutations. Follow the "
            "`pull_request / merged` and `issue / opened` playbooks for what "
            "to extract per item.\n\n"
            "PHASE 3 — explicit docs/links: if the event payload names a "
            "README, ADR, runbook, doc URL, or other authored document, read "
            "that exact source and record Document / Decision / Preference / "
            "Runbook memory only when the text explicitly supports it.\n\n"
            "Idempotent stable keys per artifact: ``activity:github:pr:"
            "<owner>/<repo>:<n>`` and ``activity:github:issue:<owner>/<repo>:"
            "<n>`` — so a backfilled artifact and a later live webhook "
            "converge instead of duplicating.\n\n"
            "SINGLE-EVENT CONTRACT: this batch contains exactly ONE event — "
            "the repository.added seed. Pass ITS event_id to EVERY "
            "apply_graph_mutations call and to the final mark_event_processed; "
            "per-artifact identity lives in the entity_keys above, not in "
            "event ids. When the todo list is drained (or you reach your "
            "budget with a coherent subset), mark_event_processed(seed) then "
            "finish_batch."
        ),
        skip=(
            "Do NOT use working-tree tools, walk the file tree, scan manifests, "
            "or infer code structure. Do NOT page or scrape beyond what the "
            "list tools return — the window/cap is deliberate. NEVER fabricate "
            "a PR, issue, document, service, dependency, feature, or preference "
            "you did not actually read — add a warning instead."
        ),
        tool_hints=(
            "github_list_pull_requests",
            "github_list_issues",
            "github_get_pull_request",
            "github_get_issue",
            "github_get_pull_request_commits",
            "github_get_pull_request_review_comments",
            "github_get_pull_request_issue_comments",
            "apply_graph_mutations",
            "mark_event_processed",
            "finish_batch",
            "web_fetch",
        ),
        max_tool_calls=400,
        enables_planner=True,
    )
)


# --- github / repository / one_shot_ingest (PR + issue backfill skill) -----
# Source of truth lives in ``domain/playbooks/repo_one_shot_ingestion.md`` so
# Claude Code and the internal reconciliation agent read the exact same prompt.
# The markdown body is embedded into ``extract`` at module import.
_register(
    EventPlaybook(
        source_system="github",
        event_type="repository",
        action="one_shot_ingest",
        summary=(
            "One-time backfill of a repository's recent merged pull requests "
            "and standalone GitHub issues into the context graph. Not "
            "incremental — live updates continue via the "
            "github/pull_request/merged and github/issue/opened webhook paths."
        ),
        available_data=(
            "Payload may include owner, repo, and count (per-kind list "
            "limit). The embedded skill below is authoritative for procedure, "
            "tool surface, key formats, and mutation shapes."
        ),
        extract=_load_skill_body("repo_one_shot_ingestion.md"),
        skip=(
            "Do NOT re-emit github/repository/added. Do NOT page past one "
            "list call per kind (one github_list_pull_requests + one "
            "github_list_issues). Do NOT read code diffs unless commit + "
            "branch + title + body + review comments all leave intent "
            "unclear. Do NOT emit Fix nodes from issue filings alone — Fix "
            "is reserved for merged PRs."
        ),
        tool_hints=(
            "github_list_pull_requests",
            "github_list_issues",
            "github_get_pull_request",
            "github_get_pull_request_commits",
            "github_get_pull_request_review_comments",
            "github_get_pull_request_issue_comments",
            "github_get_issue",
            "apply_graph_mutations",
            "mark_event_processed",
            "finish_batch",
        ),
        max_tool_calls=400,
        enables_planner=True,
    )
)


# --- linear / linear_team / added (team backfill seed) ---------------------
_register(
    EventPlaybook(
        source_system="linear",
        event_type="linear_team",
        action="added",
        summary=(
            "A Linear team was just connected to this pot. This is the "
            "initial-backfill seed for that team: the graph has no Linear "
            "history yet, so your job is to enumerate the team's existing "
            "issues and seed them into the timeline."
        ),
        available_data=(
            "Payload carries the Linear team id/name and the integration "
            "binding. Three enumerators (each bounded to a trailing window + "
            "item cap, newest-first) cover the team's history: "
            "linear_list_issues / linear_list_projects / linear_list_documents; "
            "linear_get_issue / linear_get_project / linear_get_document "
            "hydrate one ref each."
        ),
        extract=(
            "Your todo/plan tools are ON — use them as a durable worklist "
            "(it survives checkpoint resume; continue an existing list, do "
            "not re-enumerate). Enumerate ALL THREE kinds, then drain:\n"
            "  a. linear_list_projects(), linear_list_documents(), "
            "     linear_list_issues() — ONE call each returns the bounded "
            "     set of compact refs. Do not page beyond them.\n"
            "  b. Write one todo per returned ref across all three kinds. "
            "     Prefer draining projects and documents first (they frame "
            "     what the issues are about), then issues newest-first.\n"
            "  c. PROJECTS: linear_get_project(id) → seed a Feature (a "
            "     project is a unit of planned work) keyed "
            "     ``linear:project:<id>``; edge it to the issues/Decisions it "
            "     contains where evidenced; an Activity for its creation.\n"
            "  d. DOCUMENTS: linear_get_document(id) → seed a Document keyed "
            "     ``linear:document:<id>`` (title + content summary; link to "
            "     its project via RELATED_TO when present). Specs/PRDs/RFCs "
            "     may also justify a Decision.\n"
            "  e. ISSUES: linear_get_issue(ref.identifier) → an Activity "
            "     (PERFORMED + TOUCHED + IN_PERIOD); where evidenced a "
            "     Fix / Feature / Decision and edges to the work it touches; "
            "     comments are discussion context, not standalone facts. "
            "     Key ``linear:issue:<identifier>`` (e.g. "
            "     ``linear:issue:ENG-123``).\n"
            "Stable keys make a backfilled artifact and a later live Linear "
            "webhook converge instead of duplicating.\n"
            "SINGLE-EVENT CONTRACT: this batch has exactly ONE event — the "
            "linear_team.added seed. Use ITS event_id for every "
            "apply_graph_mutations and the final mark_event_processed; "
            "per-artifact identity is the entity_key, not the event id. Drain "
            "the lists (or a coherent recent subset within budget), then "
            "mark_event_processed(seed) and finish_batch."
        ),
        skip=(
            "Do NOT page past the bounded list results — the window/cap is "
            "deliberate and the tail arrives via live webhooks / future "
            "backfill. Recent coherent breadth beats exhaustive depth. NEVER "
            "invent an issue, project, document, comment, or state you did "
            "not fetch — warn instead. Do not auto-resolve open issues. If a "
            "list tool errors (e.g. the workspace has no documents API "
            "access), record a warning and continue with the kinds that did "
            "return — do not fabricate the missing kind."
        ),
        tool_hints=(
            "linear_list_projects",
            "linear_get_project",
            "linear_list_documents",
            "linear_get_document",
            "linear_list_issues",
            "linear_get_issue",
            "context_infra_topology",
            "web_fetch",
        ),
        max_tool_calls=400,
        enables_planner=True,
    )
)


# --- linear / linear_team / one_shot_ingest --------------------------------
_register(
    EventPlaybook(
        source_system="linear",
        event_type="linear_team",
        action="one_shot_ingest",
        summary=(
            "One-time ingestion of a Linear team's recent projects, documents, "
            "and issues into the context graph."
        ),
        available_data=(
            "Payload carries team and count. The embedded skill below is "
            "authoritative for the bounded list calls, drain order, key formats, "
            "and mutation shapes."
        ),
        extract=_load_skill_body("linear_team_one_shot_ingestion.md"),
        skip=(
            "Do not page past the bounded list results. Do not fabricate missing "
            "Linear projects, documents, issues, comments, or states. Do not emit "
            "Fix nodes from issue filings alone."
        ),
        tool_hints=(
            "linear_list_projects",
            "linear_get_project",
            "linear_list_documents",
            "linear_get_document",
            "linear_list_issues",
            "linear_get_issue",
            "apply_graph_mutations",
            "mark_event_processed",
            "finish_batch",
        ),
        max_tool_calls=400,
        enables_planner=True,
    )
)


# --- linear / linear_team / diff_sync --------------------------------------
_register(
    EventPlaybook(
        source_system="linear",
        event_type="linear_team",
        action="diff_sync",
        summary=(
            "Incremental catch-up ingestion of Linear projects, documents, and "
            "issues changed since the team's last successful sync cursor."
        ),
        available_data=(
            "Payload carries team, optional since, and count. The embedded "
            "skill below is authoritative for auditing existing context-graph "
            "Activity coverage before hydrating source refs, history-file "
            "handling, cursor overlap, and mutation reuse."
        ),
        extract=_load_skill_body("linear_team_diff_sync.md"),
        skip=(
            "Do not advance the cursor before graph writes succeed. Do not "
            "overwrite sync history. Do not emit Fix nodes from Linear issues."
        ),
        tool_hints=(
            "read_sync_history",
            "write_sync_history",
            "context_search",
            "context_timeline",
            "linear_list_projects",
            "linear_get_project",
            "linear_list_documents",
            "linear_get_document",
            "linear_list_issues",
            "linear_get_issue",
            "apply_graph_mutations",
            "mark_event_processed",
            "finish_batch",
        ),
        max_tool_calls=400,
        enables_planner=True,
    )
)


# --- jira / jira_project / diff_sync ---------------------------------------
_register(
    EventPlaybook(
        source_system="jira",
        event_type="jira_project",
        action="diff_sync",
        summary=(
            "Incremental catch-up ingestion of Jira epics and issues changed "
            "since the project's last successful sync cursor."
        ),
        available_data=(
            "Payload carries project_key, optional since, and count. The "
            "embedded skill below is authoritative for auditing existing "
            "context-graph Activity coverage before hydrating source refs, "
            "history-file handling, JQL updated searches, optional changelog "
            "reads, and mutation reuse."
        ),
        extract=_load_skill_body("jira_project_diff_sync.md"),
        skip=(
            "Do not advance the cursor before graph writes succeed. Do not "
            "overwrite sync history. Do not emit Fix nodes from Jira issues."
        ),
        tool_hints=(
            "read_sync_history",
            "write_sync_history",
            "context_search",
            "context_timeline",
            "jira_search_issues",
            "jira_get_issue",
            "jira_get_issue_changelog",
            "jira_bulk_fetch_changelogs",
            "apply_graph_mutations",
            "mark_event_processed",
            "finish_batch",
        ),
        max_tool_calls=400,
        enables_planner=True,
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
            "context_timeline",
            "web_fetch",
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
        tool_hints=("github_get_issue", "context_search", "web_fetch"),
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
            "The supplied text is the source of truth. If the note links to "
            "a specific URL and reading it would materially ground the "
            "Document/Decision, fetch THAT url with web_fetch — never browse "
            "open-endedly, follow chained links, or invent content the page "
            "did not contain. Don't infer a code area unless the note "
            "explicitly names one."
        ),
        tool_hints=("context_search", "web_fetch"),
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


def is_default_playbook(pb: EventPlaybook) -> bool:
    """True when ``pb`` is the generic catch-all fallback.

    Its ``tool_hints`` are advisory guidance for unclassified events, not
    an authorization boundary — callers enforcing a server-side tool
    allowlist must exclude it so a fallback event-kind is not crippled.
    """
    return pb is _DEFAULT_PLAYBOOK


def all_registered_playbooks() -> list[EventPlaybook]:
    """Return every registered playbook (introspection / docs / tests)."""
    return list(_REGISTRY.values())


def playbooks_enable_planner(playbooks: list[EventPlaybook]) -> bool:
    """True when any playbook in the batch wants the agent's planner on.

    The single declarative signal the reconciliation agent reads to decide
    whether to construct the deep agent with its todo/plan tools enabled.
    """
    return any(pb.enables_planner for pb in playbooks)


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
    "playbooks_enable_planner",
    "render_playbooks_section",
]
