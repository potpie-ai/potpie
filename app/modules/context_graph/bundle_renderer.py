"""Potpie-only: render context-engine IntelligenceBundle for LLM prompts (not part of context-engine package)."""

from __future__ import annotations

from typing import Any


def intelligence_coverage_status(bundle: dict[str, Any] | None) -> str:
    """Uppercase coverage label from bundle, or 'unknown'."""
    if not bundle or not isinstance(bundle, dict):
        return "unknown"
    cov = bundle.get("coverage") or {}
    if not isinstance(cov, dict):
        return "unknown"
    return str(cov.get("status") or "unknown").upper()


def prefetch_runtime_banner(bundle: dict[str, Any] | None) -> str:
    """
    Short block prepended *before* the rendered bundle so it wins attention order
    over generic system instructions like "use tools to gather information".
    """
    status = intelligence_coverage_status(bundle)
    if status == "COMPLETE":
        return (
            "\n<<< RUNTIME PREFETCH — EVIDENCE COMPLETE >>>\n"
            "Do NOT call duplicate graph tools: context_resolve, context_search, "
            "or ask_knowledge_graph_queries to re-fetch evidence already present here.\n"
            "Do NOT fetch .github/CODEOWNERS for ownership — use [Ownership] below; "
            "if empty, say the graph has no ownership for that path.\n"
            "Answer from CONTEXT INTELLIGENCE below first. Use fetch_file / "
            "get_code_from_probable_node_name only if you need source code not in the block.\n"
            "<<< END RUNTIME PREFETCH >>>\n\n"
        )
    if status == "PARTIAL":
        return (
            "\n<<< RUNTIME PREFETCH — EVIDENCE PARTIAL >>>\n"
            "Prefer CONTEXT INTELLIGENCE below; call only tools listed under "
            "MANDATORY TOOL-CALL RULES for missing families.\n"
            "<<< END RUNTIME PREFETCH >>>\n\n"
        )
    return ""


def supervisor_prefetch_section(bundle: dict[str, Any] | None) -> str:
    """
    Injected into multi-agent supervisor instructions when QnA has prefetched intelligence.
    Overrides generic 'delegate GitHub for PR' and 'TODO first' patterns for this turn.
    """
    status = intelligence_coverage_status(bundle)
    if status == "unknown":
        return ""

    if status == "COMPLETE":
        return """
**CONTEXT INTELLIGENCE PREFETCH (THIS TURN — OVERRIDES DEFAULT SUPERVISOR RULES):**
Evidence coverage is COMPLETE. `Additional Context` already contains PR/discussion/history/semantic hits
from the context engine.

- **Do NOT** delegate to the GitHub agent for "what happened in PR #..." / rationale / file discussion
  questions — answer from the prefetched block first.
- **Do NOT** call `context_resolve`, `context_search`, or `ask_knowledge_graph_queries`
  to re-fetch the same data.
- **Do NOT** start TODO workflows for single-turn factual Q&A that the prefetched block already answers.
- **Do NOT** fetch `CODEOWNERS` for ownership — use `[Ownership]` in the block; if absent, state that
  no ownership is recorded in the graph.
- **Use** `fetch_file` / `get_code_from_probable_node_name` / `bash_command` only when the user needs
  actual source code not present in the prefetch.

---

"""

    return """
**CONTEXT INTELLIGENCE PREFETCH (THIS TURN):**
Evidence is PARTIAL. Prefer the CONTEXT INTELLIGENCE block, then call only the tools indicated
by `MANDATORY TOOL-CALL RULES` inside it for missing families. Avoid duplicate calls for families
already present.

---

"""


def render_intelligence_bundle(bundle: dict[str, Any]) -> str:
    """
    Render a prompt-safe, human-readable block.

    Input is a JSON-serializable dict (typically dataclasses.asdict(IntelligenceBundle)).
    """
    if not bundle:
        return ""

    coverage = (bundle.get("coverage") or {}) if isinstance(bundle, dict) else {}
    status = str(coverage.get("status") or "unknown").upper()
    missing = coverage.get("missing") or []
    available = coverage.get("available") or []

    semantic_hits = bundle.get("semantic_hits") or []
    artifacts = bundle.get("artifacts") or []
    changes = bundle.get("changes") or []
    decisions = bundle.get("decisions") or []
    discussions = bundle.get("discussions") or []
    ownership = bundle.get("ownership") or []

    lines: list[str] = []
    lines.append("=== CONTEXT INTELLIGENCE (PREFETCHED) ===")

    if semantic_hits:
        lines.append("")
        lines.append("[Semantic hits]")
        for h in semantic_hits[:8]:
            name = (h or {}).get("name")
            summ = (h or {}).get("summary")
            if name and summ:
                lines.append(f"- {name}: {summ}")
            elif name:
                lines.append(f"- {name}")

    if artifacts:
        lines.append("")
        lines.append("[Artifacts]")
        for a in artifacts[:3]:
            if not isinstance(a, dict):
                continue
            kind = a.get("kind")
            ident = a.get("identifier")
            title = a.get("title")
            author = a.get("author")
            summary = a.get("summary")
            head = f"{kind}:{ident}" if kind and ident else (title or "artifact")
            meta = []
            if author:
                meta.append(f"author={author}")
            if meta:
                head = f"{head} ({', '.join(meta)})"
            lines.append(f"- {head}")
            if summary:
                lines.append(f"  - {summary}")

    if changes:
        lines.append("")
        lines.append("[Change history]")
        for c in changes[:10]:
            if not isinstance(c, dict):
                continue
            prn = c.get("pr_number")
            title = c.get("title")
            why = c.get("summary") or ""
            head = f"PR #{prn}" if prn else (c.get("artifact_ref") or "Change")
            if title:
                head = f"{head}: {title}"
            lines.append(f"- {head}")
            if why:
                lines.append(f"  - Why: {why[:240]}")

    if decisions:
        lines.append("")
        lines.append("[Decisions]")
        for d in decisions[:10]:
            if not isinstance(d, dict):
                continue
            decision = d.get("decision") or ""
            rationale = d.get("rationale") or ""
            if not decision and not rationale:
                continue
            lines.append(f"- {decision[:220]}")
            if rationale:
                lines.append(f"  - Rationale: {rationale[:260]}")

    if discussions:
        lines.append("")
        lines.append("[Discussions]")
        for t in discussions[:10]:
            if not isinstance(t, dict):
                continue
            src = t.get("source_ref") or "thread"
            fp = t.get("file_path")
            line = t.get("line")
            headline = t.get("headline") or t.get("summary") or ""
            loc = ""
            if fp:
                loc = f" @ {fp}"
                if line:
                    loc += f":{line}"
            lines.append(f"- {src}{loc}")
            if headline:
                lines.append(f"  - {headline[:260]}")

    if ownership:
        lines.append("")
        lines.append("[Ownership]")
        for o in ownership[:8]:
            if not isinstance(o, dict):
                continue
            fp = o.get("file_path")
            owner = o.get("owner")
            sig = o.get("confidence_signal")
            if fp and owner:
                tail = f" ({sig})" if sig else ""
                lines.append(f"- {fp}: {owner}{tail}")

    lines.append("")
    lines.append(f"[Evidence coverage: {status}]")
    if available:
        lines.append(f"  Available families: {', '.join(map(str, available))}")
    if missing:
        lines.append(f"  Missing families: {', '.join(map(str, missing))}")

    lines.append("")
    lines.append(">>> MANDATORY TOOL-CALL RULES (you MUST follow these) <<<")

    _tool_map = {
        "semantic_search": "context_search",
        "artifact_context": "context_resolve with PR scope",
        "change_history": "context_resolve with recent_changes include",
        "decision_context": "context_resolve with decisions include",
        "discussion_context": "context_resolve with discussions include",
        "ownership_context": "(no tool — ownership only comes from this block)",
    }

    if status == "COMPLETE":
        lines.append("Coverage is COMPLETE. All evidence needed to answer is ABOVE.")
        lines.append(
            "DO NOT call any of these tools — they will return duplicate data:"
        )
        for fam in available:
            tool = _tool_map.get(fam, "")
            if tool:
                lines.append(f"  - {fam} already fetched → skip {tool}")
        lines.append(
            "You MAY still call code-level tools (fetch_file, get_code_from_probable_node_name, "
            "analyze_code_structure, get_code_file_structure, get_node_neighbours_from_node_id) "
            "ONLY if you need to read actual source code not present above."
        )
        lines.append(
            "If the user's question can be fully answered from the evidence above, answer DIRECTLY. "
            "Do NOT start a multi-tool exploration."
        )
    elif status == "PARTIAL":
        lines.append("Coverage is PARTIAL. Some families are missing.")
        lines.append("DO NOT re-call tools for families already available:")
        for fam in available:
            tool = _tool_map.get(fam, "")
            if tool:
                lines.append(f"  - {fam} already fetched → skip {tool}")
        if missing:
            lines.append("You MAY call tools ONLY for the missing families:")
            for fam in missing:
                tool = _tool_map.get(fam, "")
                if tool:
                    lines.append(f"  - {fam} missing → call {tool} if needed")
        lines.append(
            "If the available evidence is sufficient to answer, answer directly without extra calls."
        )
    else:
        lines.append(
            "Coverage status unknown. Use your judgment and call context tools as needed."
        )

    lines.append(">>> END RULES <<<")
    lines.append("=== END CONTEXT INTELLIGENCE ===")
    return "\n".join(lines) + "\n"
